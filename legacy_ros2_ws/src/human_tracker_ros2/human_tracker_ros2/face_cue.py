"""
Face detection + recognition for the multi-cue owner matcher.

Uses OpenCV's built-in YuNet detector and SFace recogniser (both present in
cv2 >= 4.5.4). Runs on the person crop, not the whole frame, so cost scales
with the number of tracked candidates rather than image area.

The point of this cue is abstention: when the owner turns away or the face is
too small to resolve, the detector returns nothing and the cue reports
confidence 0, dropping out of the fusion entirely.
"""
import os
import cv2
import numpy as np

YUNET_PATH = os.path.expanduser("~/models/face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.expanduser("~/models/face_recognition_sface_2021dec.onnx")

# A face smaller than this (in the person crop) is not worth embedding —
# SFace expects 112x112 and upsampling a tiny crop yields noise.
MIN_FACE_PX = 24          # measured: a face at 2.5 m is ~39x53 px in a 640x480 frame
UPSCALE_TO  = 320         # small head crops are upscaled before detection
DETECT_CONF = 0.7      # YuNet score threshold
NMS_THRESH  = 0.3


class FaceCue:
    def __init__(self):
        self.enabled = False
        try:
            self.detector = cv2.FaceDetectorYN.create(
                YUNET_PATH, "", (320, 320),
                score_threshold=DETECT_CONF,
                nms_threshold=NMS_THRESH,
                top_k=5,
            )
            self.recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
            self.enabled = True
            print("[FACE] YuNet + SFace loaded")
        except Exception as e:
            print(f"[FACE] disabled: {e}")

    def detect(self, frame, bbox):
        """Return an aligned face crop from within the person bbox, or None.

        None is the abstention signal: no frontal face resolvable here.
        """
        if not self.enabled:
            return None
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = frame.shape[:2]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            if (x2 - x1) < MIN_FACE_PX or (y2 - y1) < MIN_FACE_PX:
                return None

            # Faces live in the upper portion of a standing person.
            head_y2 = y1 + int(0.40 * (y2 - y1))
            crop = frame[y1:head_y2, x1:x2]
            if crop.size == 0:
                return None

            ch, cw = crop.shape[:2]
            if ch < MIN_FACE_PX or cw < MIN_FACE_PX:
                return None

            # YuNet resolves faces poorly in a small input. Upscale the head
            # region so the face occupies enough pixels, then map the
            # detection back to the original crop for alignment.
            scale = 1.0
            det_img = crop
            if max(ch, cw) < UPSCALE_TO:
                scale = UPSCALE_TO / max(ch, cw)
                det_img = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                                     interpolation=cv2.INTER_LINEAR)

            dh, dw = det_img.shape[:2]
            self.detector.setInputSize((dw, dh))
            _, faces = self.detector.detect(det_img)
            if faces is None or len(faces) == 0:
                return None

            # Largest detection wins — the owner's own face, not a bystander
            # caught at the edge of the crop.
            face = max(faces, key=lambda f: f[2] * f[3])
            if face[2] < MIN_FACE_PX * scale or face[3] < MIN_FACE_PX * scale:
                return None

            return self.recognizer.alignCrop(det_img, face)
        except Exception:
            return None

    def embed(self, face_crop):
        """128-d L2-normalised embedding, or None."""
        if not self.enabled or face_crop is None or face_crop.size == 0:
            return None
        try:
            feat = self.recognizer.feature(face_crop)
            v = np.asarray(feat).ravel().astype(np.float32)
            n = np.linalg.norm(v) + 1e-6
            return v / n
        except Exception:
            return None
