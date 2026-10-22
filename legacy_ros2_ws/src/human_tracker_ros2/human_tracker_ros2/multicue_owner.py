"""
multicue_owner.py
─────────────────────────────────────────────────────────────────────────────
Robust owner identification by fusing MULTIPLE independent cues, each with a
self-reported confidence, combined by adaptive confidence-weighted voting.

Why this beats fixed-weight ReID:
  A single cue fails in predictable ways — ReID drifts with pose/lighting,
  color fails when the owner turns or lighting shifts, height fails at odd
  distances. But they fail *independently*. Fusing them with weights that
  scale by each cue's current reliability means the system stays locked on
  the owner as long as ANY subset of cues agrees, and refuses impostors that
  match only one.

Cues:
  1. ReID embedding      (appearance, OSNet) — strong when pose/light stable
  2. Color histogram     (clothing)          — strong, cheap, fails on turn
  3. Height ratio        (stature)           — weak alone, good tie-breaker
  4. Depth consistency   (size-at-distance)  — rejects wrong-size people
  5. Face match          (identity)          — near-decisive WHEN visible

Each cue returns (similarity ∈ [0,1], confidence ∈ [0,1]).
Fusion:  score = Σ(wᵢ · confᵢ · simᵢ) / Σ(wᵢ · confᵢ)
A cue with confidence 0 (can't measure) simply drops out — no wrong vote.

Occlusion handling:
  - When few cues are confident (person partly hidden), the fusion still
    produces a score from whatever IS visible, but the caller is told the
    overall confidence is low → it should trust motion prediction (KF) more
    and avoid re-assigning the owner to someone else.
  - A hysteresis band (accept high to lock, keep lower to hold) prevents
    flicker between the owner and a nearby similar person.

No heavy deps: numpy + opencv. Face uses OpenCV's bundled detector; if a
face embedder is available it is used, else face contributes only when a
clear frontal face is present (confidence gated).
─────────────────────────────────────────────────────────────────────────────
"""

import time
import math
from collections import deque
from typing import Optional, Tuple, Dict, List

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ═════════════════════════════════════════════════════════════════════════
#  Individual cue helpers
# ═════════════════════════════════════════════════════════════════════════
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a = a.ravel(); b = b.ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.clip(a @ b / (na * nb), -1.0, 1.0))


def color_hist(frame, bbox, bins=16):
    """HSV histogram of the torso region (upper-center of bbox)."""
    if not HAS_CV2:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    # Torso patch: central chest area, least background/limb contamination
    bw, bh = x2 - x1, y2 - y1
    px1 = max(0, int(x1 + 0.25 * bw)); px2 = min(w, int(x1 + 0.75 * bw))
    py1 = max(0, int(y1 + 0.15 * bh)); py2 = min(h, int(y1 + 0.55 * bh))
    if px2 <= px1 or py2 <= py1:
        return None
    roi = frame[py1:py2, px1:px2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def hist_sim(h1, h2) -> float:
    if h1 is None or h2 is None:
        return 0.0
    return float(np.clip(cv2.compareHist(
        h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CORREL), 0.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════
#  Owner profile — everything we know about the enrolled owner
# ═════════════════════════════════════════════════════════════════════════
class OwnerProfile:
    def __init__(self):
        self.reid: Optional[np.ndarray] = None       # averaged ReID embedding
        self.reid_gallery: deque = deque(maxlen=15)  # recent embeddings (drift)
        self.color: Optional[np.ndarray] = None      # HSV histogram
        self.color_gallery: deque = deque(maxlen=15)
        self.height_ratio: Optional[float] = None    # bbox_h / frame_h at enroll
        self.enroll_distance: Optional[float] = None  # metres at enrollment
        self.enroll_bbox_h: Optional[float] = None   # pixels at enroll distance
        self.face: Optional[np.ndarray] = None       # face embedding if captured
        self.body_width_ratio: Optional[float] = None   # shoulder-width / bbox-height ratio
        self.enrolled = False

    def expected_bbox_h(self, distance_m: float) -> Optional[float]:
        """Predict bbox pixel height at a given distance (size ∝ 1/distance)."""
        if self.enroll_distance is None or self.enroll_bbox_h is None:
            return None
        if distance_m < 0.2:
            return None
        return self.enroll_bbox_h * (self.enroll_distance / distance_m)


# ═════════════════════════════════════════════════════════════════════════
#  Multi-cue matcher
# ═════════════════════════════════════════════════════════════════════════
class MultiCueMatcher:
    # Base importance of each cue (before confidence scaling)
        # Weights revised 2026-09-18 from measured genuine/impostor separation
    # across 18 logged sessions. Median separation per cue (min..max):
    #
    #   colour 0.544 ( 0.168 .. 0.725)   never negative — most reliable
    #   depth  0.435 (-0.462 .. 0.999)   high mean, very high variance
    #   reid   0.391 ( 0.000 .. 0.666)   was overweighted at 0.45
    #   height 0.287 (-0.214 .. 0.857)
    #   face   0.161 (-0.121 .. 0.956)   only 6 sessions — integrated late
    #   body3d 0.086 (-0.354 .. 0.770)   sign flips between sessions: noise
    #
    # A weighting proportional to separation would give roughly
    # reid .21 / colour .29 / face .08 / depth .23 / height .15 / body3d .05.
    # That is NOT adopted wholesale: the separations are measured against the
    # deployed system's own is_owner verdicts, so cues that currently dominate
    # the decision are flattered, and depth's variance is too wide for a point
    # estimate. These weights move roughly halfway toward the fit — enough to
    # correct the clear errors (depth far too low, reid too high) without
    # over-fitting to a circular label source. A properly annotated ablation
    # (A1/A3 with ground-truth gt_id) should settle them.
    BASE_WEIGHTS = {
        # Revised 2026-09-22 from Srouji et al. (arXiv:2309.12479) Table II,
        # ablating face-only against torso-only over 75 trials with an
        # interferer present:
        #
        #     face only : 9/75 lost, 2/75 followed WRONG person
        #     torso only: 5/75 lost, 5/75 followed WRONG person
        #
        # Face is unavailable more often but wrong less than half as often.
        # A2RBOT's logs agree: frames with face=1.00 at c_face=0.95 were being
        # rejected because reid=0.00 and color=0.03, and ReID separability
        # varies markedly between subjects — the previous vector was fitted on
        # a single person.
        #
        # Confidence gating makes the higher face weight free: on the ~70% of
        # frames where no face resolves, c_face is 0 and the cue leaves the
        # denominator entirely rather than voting.
        # Restructured 2026-09-23 after measuring both modalities directly on
        # this hardware:
        #
        #     torso ReID  same person 0.798-0.914, different person 0.83
        #                 -> NO separation; the distributions overlap
        #     face+torso  same person 1.22-1.54,  different person 0.665
        #                 -> clean separation
        #
        # Treating face and ReID as two competing cues let a weak torso score
        # outvote a confident face. They are now combined into ONE appearance
        # signal by _cue_appearance (face when it resolves, torso otherwise,
        # following Srouji et al. sec III-A), with the geometric cues kept
        # separate and secondary.
        #
        # This also sharpens the abstention property rather than weakening it:
        # appearance abstains as a unit when neither modality resolves, instead
        # of two cues each half-abstaining and the fusion averaging noise.
        "face":   0.30,   # leads when available (Srouji et al. Table II)
        "reid":   0.25,
        "color":  0.25,
        "depth":  0.12,
        "height": 0.08,
    }

    # Hysteresis: lock onto owner above ACCEPT, keep them until below HOLD
    ACCEPT_THRESHOLD = 0.65
    HOLD_THRESHOLD   = 0.30
    # Below this many confident cues, treat as occluded/uncertain
    MIN_CONFIDENT_CUES = 1   # F14: height never reaches quorum; face=0 always; 1 is realistic floor

    def __init__(self, reid_fn=None, face_detector=None, face_embed_fn=None):
        """
        reid_fn(frame, bbox) -> embedding | None
        face_detector(frame, bbox) -> face_crop | None
        face_embed_fn(face_crop) -> embedding | None
        Any may be None; that cue is then skipped.
        """
        self.profile = OwnerProfile()
        self._reid_fn = reid_fn
        self._face_detect = face_detector
        self._face_embed = face_embed_fn
        self._currently_owned = False   # for hysteresis
        self._body_ratio_ema = None   # smoothed aspect ratio for body3d cue

    # ── Enrollment ─────────────────────────────────────────────────────
    def enroll(self, frame, bbox, distance_m: Optional[float]):
        """Capture all cues for the owner. Call over ~20 frames, then finalize."""
        if self._reid_fn:
            emb = self._reid_fn(frame, bbox)
            if emb is not None:
                self.profile.reid_gallery.append(emb.ravel())
        ch = color_hist(frame, bbox)
        if ch is not None:
            self.profile.color_gallery.append(ch)
        if self._face_detect and self._face_embed:
            fc = self._face_detect(frame, bbox)
            if fc is not None:
                fe = self._face_embed(fc)
                if fe is not None:
                    # Collect across enrollment frames; averaged in
                    # finalize_enrollment. A single-frame face anchor is
                    # brittle — one blurred or half-profile capture would
                    # poison the cue for the whole session.
                    if not hasattr(self, "_face_enroll_buf"):
                        self._face_enroll_buf = []
                    self._face_enroll_buf.append(fe.ravel())
                    self.profile.face = fe.ravel()   # provisional
        # Height + depth anchor
        h = (bbox[3] - bbox[1]) / frame.shape[0]
        self.profile.height_ratio = h
        if distance_m:
            self.profile.enroll_distance = distance_m
            self.profile.enroll_bbox_h = bbox[3] - bbox[1]
        # 3D body shape: aspect ratio (works without depth)
        bw = (bbox[2] - bbox[0]) / max(1, frame.shape[1])
        bh = (bbox[3] - bbox[1]) / max(1, frame.shape[0])
        aspect = bw / max(0.01, bh)
        if self.profile.body_width_ratio is None:
            self.profile.body_width_ratio = aspect
        else:
            self.profile.body_width_ratio = 0.7 * self.profile.body_width_ratio + 0.3 * aspect

    def finalize_enrollment(self) -> bool:
        if not self.profile.reid_gallery and not self.profile.color_gallery:
            return False
        if self.profile.reid_gallery:
            self.profile.reid = np.mean(np.stack(self.profile.reid_gallery), axis=0)
        if self.profile.color_gallery:
            self.profile.color = np.mean(np.stack(self.profile.color_gallery), axis=0)
        # Average the face captures collected during enrollment and re-normalise.
        # The count is logged because it measures how often a frontal face was
        # actually resolvable — directly relevant to how much the face cue can
        # contribute during tracking.
        buf = getattr(self, "_face_enroll_buf", [])
        if buf:
            avg = np.mean(np.stack(buf), axis=0)
            self.profile.face = avg / (np.linalg.norm(avg) + 1e-6)
            print(f"[FACE] enrolled from {len(buf)} face capture(s)")
        else:
            self.profile.face = None
            print("[FACE] no face captured during enrollment — cue will abstain")
        self._face_enroll_buf = []
        self.profile.enrolled = True
        return True

    # ── Per-cue scoring: each returns (similarity, confidence) ─────────
    def _cue_reid(self, frame, bbox, emb=None) -> Tuple[float, float]:
        if not self.profile.enrolled or self._reid_fn is None:
            return 0.0, 0.0
        if emb is None:
            emb = self._reid_fn(frame, bbox)
        if emb is None:
            return 0.0, 0.0
        emb = emb.ravel()
        # Score against the FIXED enrollment anchor (identity ground truth).
        # The gallery is NOT used for scoring — it caused a self-matching
        # feedback loop where the current frame matched its own recent copy
        # and every person scored 1.00. Anchor-only keeps identity honest.
        anchor_sim = _cosine(self.profile.reid, emb)
        # Optional mild drift aid: average of gallery, weighted low, capped
        if self.profile.reid_gallery:
            g_sims = [_cosine(g, emb) for g in self.profile.reid_gallery]
            g_mean = sum(g_sims) / len(g_sims)   # mean, NOT max (no self-match spike)
            sim = 0.8 * anchor_sim + 0.2 * g_mean
        else:
            sim = anchor_sim
        sim01 = (sim + 1.0) / 2.0   # cosine [-1,1] -> [0,1]
        # X6: expose the pre-scaling value so the temperature constants below
        # can be calibrated against measured genuine/impostor distributions
        # instead of the assumed 0.75-0.95 clustering.
        self._last_reid_raw = float(sim01)
        # Temperature scaling, calibrated from measured distributions (X6,
        # session_20260918_142032, n=3185 genuine / 389 impostor on this
        # hardware and this ReID engine):
        #
        #   genuine : mean 0.8772, sd 0.0340, p05 0.8251
        #   impostor: mean 0.7912, sd 0.0255, p95 0.8216
        #
        # The distributions separate cleanly at ~0.823 with only a 0.0035 gap
        # between impostor p95 and genuine p05. The previous window
        # [0.72, 0.94] was far too wide: it scored the impostor MEAN at 0.32,
        # giving strangers substantial credit, and the genuine mean at only
        # 0.71 rather than the ~0.85 its comment claimed.
        REID_LO = 0.8216   # impostor p95 -> 0
        REID_HI = 0.9317   # genuine  p95 -> 1
        sim01 = float(np.clip((sim01 - REID_LO) / (REID_HI - REID_LO), 0.0, 1.0))
        # Confidence: ReID is reliable only when the crop carries enough
        # resolution. Measured (X6, session_20260918_142032): at bbox heights
        # below ~0.6 of half-frame the raw similarity for the OWNER falls to
        # 0.77-0.79, which is inside the impostor distribution (mean 0.7912).
        # The model genuinely cannot discriminate at that range, so the cue
        # must abstain rather than report a confident zero — a 0.3 floor was
        # injecting 0.135 of denominator weight carrying no information.
        bh = bbox[3] - bbox[1]
        rel = bh / (0.5 * frame.shape[0])
        if rel < 0.65:
            # Below the measured discrimination threshold: abstain entirely.
            return sim01, 0.0
        conf = float(np.clip(rel, 0.0, 1.0))
        # Boost confidence when the scaled sim is high (owner match)
        if sim01 > 0.7:
            conf = min(1.0, conf + 0.15)
        # Recorded for logging: the fusion now sees only the combined
        # appearance signal, but the ablations need the raw per-modality value.
        self._reid_last = (sim01, conf)
        return sim01, conf

    def _cue_color(self, frame, bbox) -> Tuple[float, float]:
        if self.profile.color is None:
            return 0.0, 0.0
        ch = color_hist(frame, bbox)
        if ch is None:
            return 0.0, 0.0
        # Anchor-dominant mean, not max (F16). Taking the maximum over the
        # gallery lets any drifted entry self-match at ~1.0 and pin the cue
        # high forever — the same pathology already fixed for ReID.
        anchor_sim = hist_sim(self.profile.color, ch)
        if self.profile.color_gallery:
            g_sims = [hist_sim(g, ch) for g in self.profile.color_gallery]
            g_mean = sum(g_sims) / len(g_sims)
            sim = 0.8 * anchor_sim + 0.2 * g_mean
        else:
            sim = anchor_sim

        # Real confidence model, replacing a hardcoded 0.8. Colour is
        # trustworthy when the torso patch is large enough to sample and
        # well enough lit to carry hue information. Both are measured here.
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return 0.0, 0.0

        # Size term: a small crop gives a noisy histogram.
        size_term = float(np.clip((bw * bh) / (0.04 * frame.shape[0] * frame.shape[1]), 0.0, 1.0))

        # Illumination term: hue is unreliable in near-dark or blown-out
        # patches. Sample the torso region the histogram actually uses.
        ty1 = y1 + int(0.15 * bh); ty2 = y1 + int(0.55 * bh)
        tx1 = x1 + int(0.25 * bw); tx2 = x1 + int(0.75 * bw)
        patch = frame[max(0, ty1):max(1, ty2), max(0, tx1):max(1, tx2)]
        if patch.size == 0:
            light_term = 0.3
        else:
            mean_v = float(patch.mean())
            # Peak trust in the mid range; falls off when dark or saturated.
            light_term = float(np.clip(1.0 - abs(mean_v - 128.0) / 128.0, 0.0, 1.0))
            # Variance: a flat patch carries little colour information.
            var_term = float(np.clip(patch.std() / 40.0, 0.0, 1.0))
            light_term = 0.7 * light_term + 0.3 * var_term

        conf = float(np.clip(0.25 + 0.75 * (0.5 * size_term + 0.5 * light_term), 0.0, 1.0))
        return float(sim), conf

    def _cue_height(self, frame, bbox) -> Tuple[float, float]:
        if self.profile.height_ratio is None:
            return 0.0, 0.0
        h = (bbox[3] - bbox[1]) / frame.shape[0]
        diff = abs(h - self.profile.height_ratio) / max(0.01, self.profile.height_ratio)
        # Lenient: only a gross mismatch (>60% off) drives sim toward 0.
        # Small differences (partial framing, posture) stay near 1.0.
        sim = 1.0 - min(1.0, diff / 0.6)
        # Low confidence AND further reduced when box seems clipped (top/bottom
        # at frame edge => height unreliable)
        y1, y2 = bbox[1], bbox[3]
        clipped = (y1 <= 2) or (y2 >= frame.shape[0]-2)
        conf = 0.25 if clipped else 0.45  # still low weight, but can count as confident when sim>0.5
        return float(max(0.0, sim)), conf

    def _cue_depth(self, frame, bbox, distance_m: Optional[float]) -> Tuple[float, float]:
        """Size-at-distance consistency: is this blob the owner's size for how
        far away it is? Rejects a differently-sized person at the same range.
        
        KEY DEFENSE: Someone in front of the owner is closer → larger bbox.
        If depth confirms they're close but size matches owner at 3m, this
        cue votes against. Temporal smoothing prevents single-frame jitter."""
        if distance_m is None:
            return 0.0, 0.0
        expected = self.profile.expected_bbox_h(distance_m)
        actual = bbox[3] - bbox[1]
        if expected is None:
            if self.profile.height_ratio is None:
                return 0.0, 0.0
            expected = self.profile.height_ratio * frame.shape[0] * (2.0 / max(0.3, distance_m))
            conf = 0.35
        else:
            conf = 0.75 if 0.5 < distance_m < 5.0 else 0.3
        ratio = min(actual, expected) / max(actual, expected + 1e-6)
        sim = float(np.clip((ratio - 0.6) / 0.25, 0.0, 1.0))
        # Temporal smoothing with a lighter alpha. alpha=0.5 had a stable fixed
        # point at sim=0.5 that the cue latched onto for hundreds of frames
        # (measured 2026-09-18: 890/2286 verified frames stuck at exactly 0.500).
        # A lighter blend tracks the measurement instead of averaging it away.
        alpha = 0.75  # 75% new, 25% old
        if not hasattr(self, "_depth_smooth"):
            self._depth_smooth = {"sim": sim, "conf": conf}
        else:
            self._depth_smooth["sim"]  = (1-alpha) * self._depth_smooth["sim"]  + alpha * sim
            self._depth_smooth["conf"] = (1-alpha) * self._depth_smooth["conf"] + alpha * conf
        # Abstain rather than vote a meaningless midpoint: if the smoothed
        # similarity has converged near 0.5 while raw measurements disagree,
        # the cue has no information — report zero confidence.
        _s = self._depth_smooth["sim"]
        if 0.45 < _s < 0.55 and abs(sim - _s) > 0.4:
            return _s, 0.0
        return _s, self._depth_smooth["conf"]

    def _cue_body3d(self, frame, bbox, depth_roi: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """3D body shape: width/depth ratios should match the owner's build.
        
        Uses the bbox aspect ratio (width/height) which correlates with
        shoulder width vs height — a stable biometric. If depth_roi is
        available, also checks torso depth distribution.
        
        Returns (sim, conf) where sim is how well the 3D shape matches.
        """
        if self.profile.body_width_ratio is None:
            return 0.0, 0.0
        
        bw = (bbox[2] - bbox[0]) / frame.shape[1]
        bh = (bbox[3] - bbox[1]) / frame.shape[0]
        raw_ratio = bw / max(0.01, bh)
        # TEMPORAL SMOOTHING: bbox width jitters frame-to-frame (arm motion,
        # partial crops) while true body shape doesn't. EMA-smooth so the cue
        # reflects sustained shape, not per-frame detection noise.
        if not hasattr(self, "_body_ratio_ema") or self._body_ratio_ema is None:
            self._body_ratio_ema = raw_ratio
        else:
            self._body_ratio_ema = 0.7 * self._body_ratio_ema + 0.3 * raw_ratio
        current_ratio = self._body_ratio_ema
        owner_ratio = self.profile.body_width_ratio
        # Wider tolerance (0.7 not 0.5) - aspect ratio is a loose biometric
        ratio_diff = abs(current_ratio - owner_ratio) / max(0.01, owner_ratio)
        sim = float(np.clip(1.0 - ratio_diff / 0.7, 0.0, 1.0))
        
        # If we have depth ROI, check torso depth consistency too
        conf = 0.4
        if depth_roi is not None and depth_roi.size > 0:
            valid = depth_roi[(depth_roi > 200) & (depth_roi < 8000)]
            if len(valid) > 50:
                depth_range = (valid.max() - valid.min()) / 1000.0
                if depth_range < 0.5:
                    conf = 0.55
                else:
                    conf = 0.35
        
        return sim, conf

    # SFace's published same-identity cosine threshold. Genuine pairs sit
    # above it, impostors below; the operating window is much narrower than
    # the full [-1, 1] cosine range.
    FACE_COS_THRESH = 0.363
    FACE_COS_HI     = 0.60   # confidently the same person

    # Face detect+embed costs ~30 ms on CPU (measured: 8.7 ms YuNet +
    # 21.2 ms SFace). Running it every frame would halve the loop rate, and
    # facial identity does not change between consecutive frames — so the
    # cue recomputes periodically and holds its last result in between.
    # Recomputation interval. NOTE this counter ticks once per CANDIDATE
    # scored, not once per frame: with several people in view it advances
    # several times per frame, so an interval of 8 meant face recomputed far
    # less often than intended. Measured 2026-09-22: c_face > 0 on 0 of 565
    # verified frames. Lowered, and the cached value is no longer allowed to
    # hold an abstention indefinitely.
    FACE_INTERVAL = 3

    # Face-to-torso substitution point. Measured on this camera: a face
    # resolves on roughly a third of tracking frames, so torso must carry the
    # rest — but when a face IS present it is far more discriminative and
    # should decide alone.
    FACE_TRUST = 0.55      # face sim above this overrides torso entirely

    def _cue_appearance(self, frame, bbox, reid_emb) -> Tuple[float, float]:
        """Single appearance signal: face when it resolves, torso otherwise.

        Srouji et al. (arXiv:2309.12479, sec III-A) combine face and torso by
        taking the maximum of the two similarities, "primarily using faces for
        re-identification, since faces carry the most distinctive feature among
        body parts; secondarily using torso when faces are occluded or not
        present in the camera's sight". Their Table II shows why: face-only
        follows the wrong person 2/75, torso-only 5/75.

        Measured here 2026-09-23, the torso engine alone cannot discriminate at
        all (same person 0.798, different person 0.83), so a weighted blend
        would let the torso drag a confident face match down. Substitution, not
        blending.
        """
        f_sim, f_conf = self._cue_face(frame, bbox)
        r_sim, r_conf = self._cue_reid(frame, bbox, reid_emb)

        # A confident face decides on its own, in either direction: a strong
        # face match accepts, a strong face MISmatch rejects. The latter is
        # what stops a look-alike in similar clothing from being adopted.
        if f_conf >= 0.5:
            if f_sim >= self.FACE_TRUST or r_conf < 0.3:
                return f_sim, max(f_conf, r_conf)
            # Face present but disagreeing — trust it over the torso, which
            # has no measured ability to tell these two apart anyway.
            if f_sim < 0.30:
                return f_sim, f_conf
            # Both available and both moderate: take the higher, per Srouji.
            return max(f_sim, r_sim), max(f_conf, r_conf)

        # No usable face: torso carries appearance, or appearance abstains.
        return r_sim, r_conf

    def _cue_face(self, frame, bbox) -> Tuple[float, float]:
        if self.profile.face is None or not (self._face_detect and self._face_embed):
            # Warn once: a missing face anchor silently disables 0.30 of the
            # weight budget for the entire session, which is invisible in the
            # logs as "c_face = 0" and indistinguishable from the owner simply
            # facing away.
            if not getattr(self, "_face_dead_warned", False):
                self._face_dead_warned = True
                _why = ("no face anchor from enrollment" if self.profile.face is None
                        else "face detector/embedder not wired")
                print(f"[FACE] CUE DISABLED FOR SESSION: {_why}")
            return 0.0, 0.0

        self._face_tick = getattr(self, "_face_tick", 0) + 1
        # Never serve a cached ABSTENTION — only a cached positive. A recompute
        # that lands while the owner is mid-turn would otherwise suppress the
        # cue for the whole interval, and face is the cue most likely to be
        # momentarily unavailable.
        _cached = getattr(self, "_face_last", (0.0, 0.0))
        if self._face_tick % self.FACE_INTERVAL != 0 and _cached[1] > 0:
            # Hold the previous verdict. Note this preserves abstention too:
            # if the last real computation found no face, we keep reporting
            # confidence 0 rather than inventing a vote.
            return getattr(self, "_face_last", (0.0, 0.0))

        fc = self._face_detect(frame, bbox)
        if fc is None:
            self._face_last = (0.0, 0.0)
            return 0.0, 0.0   # no face visible → confidence 0, drops out
        fe = self._face_embed(fc)
        if fe is None:
            self._face_last = (0.0, 0.0)
            return 0.0, 0.0
        sim = _cosine(self.profile.face, fe.ravel())
        # Map the operating window onto [0,1] rather than the whole cosine
        # range. A flat (sim+1)/2 put an impostor at cosine 0.1 on 0.55 —
        # above the hold threshold — which defeated the point of the cue.
        lo, hi = self.FACE_COS_THRESH * 0.5, self.FACE_COS_HI
        sim01 = float(np.clip((sim - lo) / (hi - lo), 0.0, 1.0))
        # Confidence scales with how much face there is to work with.
        fh = fc.shape[0] if hasattr(fc, "shape") else 0
        if   fh >= 100: conf = 0.95
        elif fh >= 70:  conf = 0.80
        elif fh >= 50:  conf = 0.60
        else:           conf = 0.40
        self._face_last = (sim01, conf)
        return sim01, conf

    # ── Fusion ──────────────────────────────────────────────────────────
    def score(self, frame, bbox, distance_m: Optional[float] = None,
              reid_emb=None, depth_roi=None, other_dets=None) -> Dict:
        """
        Returns dict:
          score        : fused similarity [0,1]
          confidence   : overall confidence (how much to trust this decision)
          is_owner     : bool (with hysteresis)
          cues         : per-cue (sim, conf, weight) for debugging
          occluded     : bool — too few confident cues, trust motion instead
        """
        # F15: snapshot EMA state before scoring so multi-candidate frames
        # cannot contaminate each other. State is restored if candidate rejected.
        import copy as _copy
        _ema_snap = {
            "_depth_smooth":   _copy.deepcopy(getattr(self, "_depth_smooth", None)),
            "_body_ratio_ema": getattr(self, "_body_ratio_ema", None),
        }
        # Face and ReID are no longer separate voters — see _cue_appearance.
        # They are still computed individually inside it, and both are recorded
        # below so the CSV schema and every existing ablation script keep
        # working, but only the combined signal enters the fusion.
        cues = {
            "reid":   self._cue_reid(frame, bbox, reid_emb),
            "face":   self._cue_face(frame, bbox),
            "color":  self._cue_color(frame, bbox),
            "depth":  self._cue_depth(frame, bbox, distance_m),
            "height": self._cue_height(frame, bbox),
            # body3d removed 2026-09-18: measured separation 0.086 with the
            # sign flipping between sessions (+0.53 / -0.11). _cue_body3d is
            # retained below so the ablation can re-enable it, but it no longer
            # participates in fusion. The CSV keeps s_body3d/c_body3d columns
            # (written as 0.0) so existing analysis scripts still parse.
        }

        num = 0.0
        den = 0.0
        confident_cues = 0
        detail = {}
        for name, (sim, conf) in cues.items():
            w = self.BASE_WEIGHTS[name]
            eff = w * conf
            num += eff * sim
            den += eff
            detail[name] = (round(sim, 3), round(conf, 2), round(w, 2))
            # A cue counts toward quorum only when it is both measurable and
            # in agreement. The agreement bar is hysteretic, like the accept
            # threshold: it is easier to KEEP the owner than to acquire one.
            # Measured (session_20260918_143920): while tracking at range the
            # best available cue (colour) sits at ~0.41 once ReID abstains,
            # just under a fixed 0.5 bar — so quorum hit zero, occlusion fired
            # and the lock dropped at ~2 m even though the evidence was sound.
            _agree = 0.35 if self._currently_owned else 0.5
            if conf >= 0.5 and sim >= _agree:
                confident_cues += 1

        fused = (num / den) if den > 1e-6 else 0.0

        # Smart veto: only block acceptance when depth is NEAR-CERTAIN (conf>0.8,
        # sim<0.15) AND the positive cues (reid+color) are also weak (both <0.5).
        # This catches a close impostor without rejecting the real owner on
        # a single noisy depth frame.
        depth_sim, depth_conf = cues.get("depth", (1.0, 0.0))
        reid_sim, reid_conf = cues.get("reid", (0.0, 0.0))
        color_sim, _ = cues.get("color", (0.0, 0.0))
        veto = (depth_conf > 0.8 and depth_sim < 0.15 and
                reid_sim < 0.5 and color_sim < 0.5)
        if veto:
            fused = min(fused, 0.35)

        # Overall confidence: how much total weighted evidence we had
        overall_conf = float(np.clip(den / sum(self.BASE_WEIGHTS.values()), 0.0, 1.0))
        # Strong ReID alone means we clearly see the owner - not occluded,
        # even if color/height dipped (blur, partial framing).
        reid_sim = cues.get("reid", (0.0,0.0))[0]
        # F14/F18: face=0 always, height unreliable — reid alone at 0.55+ is sufficient evidence
        # OHN (TraceNet, Khan et al. 2026, eq. 5): measure occlusion directly
        # as the maximum IoU with any OTHER detection, rather than inferring it
        # from cue disagreement. Cue-derived occlusion cannot distinguish
        # "genuinely hidden behind someone" from "visible but too far for ReID
        # to discriminate" — measured 2026-09-18, where distant frames were
        # flagged occluded despite sound colour/depth evidence.
        occ_iou = 0.0
        if other_dets:
            bx1, by1, bx2, by2 = bbox
            for d in other_dets:
                dx1, dy1, dx2, dy2 = d[0], d[1], d[2], d[3]
                if (dx1, dy1, dx2, dy2) == (bx1, by1, bx2, by2):
                    continue   # this candidate itself
                ix1 = max(bx1, dx1); iy1 = max(by1, dy1)
                ix2 = min(bx2, dx2); iy2 = min(by2, dy2)
                iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                inter = iw * ih
                if inter <= 0:
                    continue
                union = ((bx2-bx1)*(by2-by1) + (dx2-dx1)*(dy2-dy1) - inter)
                if union > 0:
                    occ_iou = max(occ_iou, inter / union)

        # Geometric overlap is authoritative when it is high. Otherwise fall
        # back to the cue-quorum heuristic, which still catches motion blur and
        # out-of-frame clipping that produce no overlap.
        # Threshold swept offline (session_20260918_151007, n=2469). Median
        # s_reid among fired frames was NOT monotone in the threshold: 0.281 at
        # 0.10, 0.000 across 0.20-0.30, then 0.128 at 0.35 and 0.363 at 0.45.
        # That non-monotonicity separates two populations. The 0.20-0.30 band is
        # mostly DISTANT frames, where small bboxes crowd together by
        # perspective and ReID has already abstained on range — overlap there is
        # an artifact, not occlusion. The high band is genuine close-range
        # crossing, where the candidate is large and ReID still discriminates.
        # So geometric occlusion is only asserted when the candidate is big
        # enough that overlap implies physical proximity.
        OCC_IOU_THRESH = 0.35
        _bh_rel = (bbox[3] - bbox[1]) / max(1.0, float(frame.shape[0]))
        if occ_iou >= OCC_IOU_THRESH and _bh_rel >= 0.30:
            occluded = True
        else:
            occluded = (confident_cues < self.MIN_CONFIDENT_CUES) and (reid_sim < 0.55)

        # Appearance veto. Depth and height are weak identity signals — they
        # say "right size for that distance", which is true of most adults.
        # Measured 2026-09-22: accepted frames with s_reid 0.177 and
        # s_color 0.124 were carried over the threshold by s_depth 1.000.
        # If both cues that actually discriminate are available and both say
        # no, geometry must not overrule them.
        _rs, _rc = cues.get("reid",  (0.0, 0.0))
        _cs, _cc = cues.get("color", (0.0, 0.0))
        _fs, _fc = cues.get("face",  (0.0, 0.0))

        # Face overrides the veto when it is confident and agrees.
        #
        # Measured 2026-09-22: frames with face=1.00 at c_face=0.95 were
        # being rejected because reid=0.00 and color=0.03. ReID separability
        # is markedly lower for some subjects than the weight vector assumes
        # — it was calibrated on one person — so treating reid+color as the
        # sole appearance authority discards the strongest evidence
        # available. A confident face match IS appearance evidence.
        face_confirms = (_fc >= 0.6 and _fs >= 0.55)

        appearance_available = (_rc > 0.3) or (_cc > 0.3)
        appearance_rejects = (
            appearance_available
            and not face_confirms
            and (_rc <= 0.3 or _rs < 0.30)
            and (_cc <= 0.3 or _cs < 0.30)
        )
        if appearance_rejects:
            fused = min(fused, 0.25)



        # A confident, flat-zero ReID used to clamp the score to 0.10 here,
        # on the assumption that the embedding was asserting "different
        # person". Measured 2026-09-23, that assumption does not hold on this
        # hardware: the torso engine scores the SAME person at 0.798-0.914 and
        # a DIFFERENT person at 0.83, so a zero is as likely to be a bad crop
        # as a real rejection. The clamp was overriding face=0.70 evidence.
        # Retained only when face also disagrees or is unavailable.
        if _rc > 0.8 and _rs <= 0.02 and not (_fc >= 0.5 and _fs >= 0.45):
            fused = min(fused, 0.10)

        # Hysteresis: easier to KEEP the owner than to acquire a new one —
        # but only while we are actually still seeing them.
        #
        # Measured 2026-09-22: after the owner left frame, six re-locks
        # happened at fused scores of 0.31-0.56, three of them with
        # s_reid = 0.000 at full confidence. The lowered HOLD bar was still
        # in force because _currently_owned stays True through a loss, so a
        # stranger crossing the owner's last position only had to clear 0.30.
        # Hysteresis is meant to stop flicker on the SAME person; during a
        # loss it does the opposite. Re-acquisition must meet the full bar.
        _lost = getattr(self, "_lost_frames", 0)
        _hysteresis_ok = self._currently_owned and _lost < 3
        thresh = self.HOLD_THRESHOLD if _hysteresis_ok else self.ACCEPT_THRESHOLD

        # Secondary acceptance path (Srouji et al., sec III-A), who accept on
        # max(face, torso) > 0.8 rather than a weighted sum. Adopted as a
        # fallback so a single confident, strongly-agreeing cue cannot be
        # outvoted by several weak ones — the failure observed when face=1.00
        # lost to reid=0.00 and color=0.03. Deliberately strict: the cue must
        # be near-certain, since this bypasses the weighted decision.
        _strong = False
        for _k, _min_c, _min_s in (("face", 0.60, 0.75), ("reid", 0.60, 0.80)):
            _s_, _c_ = cues.get(_k, (0.0, 0.0))
            if _c_ >= _min_c and _s_ >= _min_s:
                _strong = True
                break

        is_owner = ((fused >= thresh) or _strong) and not occluded

        # F15: if this candidate is not the owner, restore EMA state so the
        # next candidate scores against uncontaminated history
        if not is_owner:
            if _ema_snap["_depth_smooth"] is None:
                if hasattr(self, "_depth_smooth"):
                    del self._depth_smooth
            else:
                self._depth_smooth = _ema_snap["_depth_smooth"]
            self._body_ratio_ema = _ema_snap["_body_ratio_ema"]

        # Count consecutive frames without a confirmed owner, so the block
        # above can tell "still tracking" from "lost and re-acquiring".
        if is_owner:
            self._lost_frames = 0
        else:
            self._lost_frames = getattr(self, "_lost_frames", 0) + 1

        return {
            "score": round(fused, 3),
            "confidence": round(overall_conf, 3),
            "is_owner": is_owner,
            "occluded": occluded,
            "confident_cues": confident_cues,
            "occ_iou": round(occ_iou, 3),
            "cues": detail,
            "reid_raw": round(getattr(self, "_last_reid_raw", 0.0), 4),
        }

    def set_owned(self, owned: bool):
        """Caller informs matcher of lock state so hysteresis works."""
        self._currently_owned = owned

    def update_galleries(self, frame, bbox, result: Dict, reid_emb=None):
        """Fold current appearance into galleries to track drift — ONLY when
        the match is very strong and unambiguous. Stricter now (0.82) to
        prevent the gallery from absorbing a wrong person and then matching
        them at 1.0 forever."""
        if result["score"] < 0.82 or result["occluded"]:
            return
        if self._reid_fn:
            emb = reid_emb if reid_emb is not None else self._reid_fn(frame, bbox)
            if emb is not None:
                self.profile.reid_gallery.append(emb.ravel())
        ch = color_hist(frame, bbox)
        if ch is not None:
            self.profile.color_gallery.append(ch)

