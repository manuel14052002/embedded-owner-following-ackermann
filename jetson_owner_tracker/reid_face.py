"""Face + torso appearance backend (Srouji et al., arXiv:2309.12479).

Why this exists
---------------
The torso-only Re-ID engine on this platform does not discriminate. Measured
2026-09-23 with ``reid.engine`` (osnet_x0_25, 512-d):

    same person, frames 3 s apart : 0.798, 0.804, 0.914
    different person              : 0.83

Self-similarity is *lower* than impostor similarity — a pose change costs more
than a change of identity. No threshold separates those distributions, which is
why threshold tuning could not fix the wrong-person re-identification.

Srouji et al. Table II ablates the two modalities over 75 trials with an
interferer present:

    face only  : 9/75 lost, 2/75 followed the WRONG person
    torso only : 5/75 lost, 5/75 followed the WRONG person
    both       : 3/75 lost, 1/75

Face is unavailable more often but wrong less than half as often, so the two
are complementary: "primarily using faces for re-identification, since faces
carry the most distinctive feature among body parts; secondarily using torso
for re-identification when faces are occluded or not present in the camera's
sight" (sec III-A).

Design
------
This class returns a **concatenated** embedding, not a fused scalar, because
the :class:`PersonReID` interface promises embeddings whose dot product is a
similarity. Layout::

    [ face_block (D_f) | torso_block (D_t) | face_present_flag ]

Both blocks are scaled so that the dot product of two embeddings reproduces
Srouji's ``max(face_sim, torso_sim)`` rule closely enough for the association
scorer, while degrading gracefully to torso-only when either side has no face.
See :meth:`_assemble` for the arithmetic.

The flag channel lets callers tell whether a face contributed, which matters
for logging and for the abstention accounting in the thesis.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from owner_tracking.config import ReIDConfig
from owner_tracking.reid import PersonReID, l2_normalize

LOGGER = logging.getLogger(__name__)

# SFace's published same-identity cosine threshold. Genuine pairs sit above it,
# impostors below; the usable window is far narrower than the full [-1, 1]
# cosine range, so similarities are remapped onto it rather than used raw.
SFACE_COS_LO = 0.30
SFACE_COS_HI = 0.60

# YuNet resolves faces poorly in a small input. A head crop is upscaled to this
# size on its long edge before detection. Measured on this camera: a face at
# 2.5 m is 39x53 px in a 640x480 frame, which YuNet finds in the full frame but
# misses in an unscaled 80x100 head crop.
UPSCALE_TO = 320

# A face below this width (in the upscaled crop) carries too little detail.
MIN_FACE_PX = 24

# Fraction of the person box searched for a head.
HEAD_FRACTION = 0.45


class FaceTorsoReID(PersonReID):
    """Face (YuNet + SFace) with torso fallback."""

    def __init__(self, config: ReIDConfig, device: str) -> None:
        self.config = config
        self.device = device

        yunet = getattr(config, "face_detector_path", "") or \
            "/home/a2r/models/face_detection_yunet_2023mar.onnx"
        sface = getattr(config, "face_embedder_path", "") or \
            "/home/a2r/models/face_recognition_sface_2021dec.onnx"

        self._detector = cv2.FaceDetectorYN.create(
            yunet, "", (UPSCALE_TO, UPSCALE_TO),
            score_threshold=0.7, nms_threshold=0.3, top_k=5)
        self._recognizer = cv2.FaceRecognizerSF.create(sface, "")

        # Torso backend. Built through the factory so the configured fallback
        # chain still applies; a failure here is not fatal because face alone
        # can carry identity, but tracking would then drop whenever the owner
        # turns away.
        self._torso = self._build_torso(config, device)
        self._torso_dim = int(getattr(self._torso, "embedding_dim", 0) or 512)
        self._face_dim = 128            # SFace feature length

        self.embedding_dim = self._face_dim + self._torso_dim + 1
        self.faces_found = 0
        self.calls = 0

        LOGGER.info("FaceTorsoReID ready: face %dd + torso %dd (+flag) = %dd",
                    self._face_dim, self._torso_dim, self.embedding_dim)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_torso(config: ReIDConfig, device: str) -> PersonReID:
        from owner_tracking.factory import _REID_BACKENDS

        for name in ("tensorrt", "torchreid", "torchvision", "histogram"):
            factory = _REID_BACKENDS.get(name)
            if factory is None or name == "face_torso":
                continue
            try:
                model = factory(config, device)
                LOGGER.info("FaceTorsoReID torso backend: %s", name)
                return model
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("torso backend '%s' unavailable: %s", name, exc)
        raise RuntimeError("FaceTorsoReID could not build any torso backend")

    # ------------------------------------------------------------------
    def _face_embedding(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """SFace feature for the head region of ``bbox``, or None."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None

        head = frame[y1:y1 + max(1, int(HEAD_FRACTION * (y2 - y1))), x1:x2]
        if head.size == 0:
            return None

        ch, cw = head.shape[:2]
        scale = 1.0
        det_img = head
        if max(ch, cw) < UPSCALE_TO:
            scale = UPSCALE_TO / max(ch, cw)
            det_img = cv2.resize(head, (int(cw * scale), int(ch * scale)),
                                 interpolation=cv2.INTER_LINEAR)

        dh, dw = det_img.shape[:2]
        self._detector.setInputSize((dw, dh))
        try:
            _, faces = self._detector.detect(det_img)
        except cv2.error:
            return None
        if faces is None or len(faces) == 0:
            return None

        face = max(faces, key=lambda f: f[2] * f[3])
        if face[2] < MIN_FACE_PX * scale:
            return None

        try:
            aligned = self._recognizer.alignCrop(det_img, face)
            feat = self._recognizer.feature(aligned)
        except cv2.error:
            return None
        feat = np.asarray(feat, dtype=np.float32).ravel()
        n = np.linalg.norm(feat)
        return feat / n if n > 1e-9 else None

    # ------------------------------------------------------------------
    def _assemble(self, face: Optional[np.ndarray], torso: np.ndarray) -> np.ndarray:
        """Pack face and torso into one vector.

        The dot product of two assembled vectors is

            w_f * cos(face_a, face_b) + w_t * cos(torso_a, torso_b)

        with the face term present only when *both* sides had a face. Weights
        are chosen so that a face agreement dominates: face carries 0.75 of the
        similarity when available, torso 0.25. When either side lacks a face the
        face block is zero and torso carries everything, which is exactly
        Srouji's fallback behaviour without needing a separate code path in the
        association scorer.

        This is a close approximation of max(face, torso) rather than the
        literal maximum, because the interface requires a fixed-length
        embedding compared by dot product. It preserves the property that
        matters: a confident face match cannot be outvoted by a weak torso
        match, and vice versa.
        """
        out = np.zeros(self.embedding_dim, dtype=np.float32)
        if face is not None:
            out[:self._face_dim] = face * np.sqrt(0.75, dtype=np.float32)
            out[-1] = 1.0
            torso_w = np.sqrt(0.25, dtype=np.float32)
        else:
            # No face: torso must carry the full similarity, otherwise a
            # face-less frame could never reach the same score as a face-ful
            # one and the owner would appear to "fade" on turning away.
            torso_w = 1.0
        out[self._face_dim:self._face_dim + self._torso_dim] = torso * torso_w
        return out

    # ------------------------------------------------------------------
    def extract_embeddings(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        if len(bboxes) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        torso = self._torso.extract_embeddings(frame, bboxes)
        out = np.zeros((len(bboxes), self.embedding_dim), dtype=np.float32)

        for i, bbox in enumerate(bboxes):
            self.calls += 1
            if not np.any(torso[i]):
                continue          # invalid crop: interface says leave zeros
            face = self._face_embedding(frame, bbox)
            if face is not None:
                self.faces_found += 1
            out[i] = self._assemble(face, torso[i])

        # Rows are NOT re-normalised to unit length: the weighting above is
        # what makes the dot product behave like the intended fusion, and
        # normalising would destroy it. Each row already has norm <= 1.
        return out

    # ------------------------------------------------------------------
    def face_rate(self) -> float:
        """Fraction of crops in which a face resolved — useful for logging."""
        return self.faces_found / max(1, self.calls)
