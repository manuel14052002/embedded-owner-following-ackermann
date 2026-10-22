"""
Person Tracker - YOLOv8 TensorRT + ReID + owner multi-cue matching
Multi-cue: ReID embedding + color histogram + height ratio
"""
import cv2
import numpy as np
import time
from typing import Optional, Dict
from ultralytics import YOLO
import onnxruntime as ort

from .owner_tracking_upgrade import KalmanCV2D, AdaptiveGallery, RecaptureManager
from .multicue_owner import MultiCueMatcher
from .face_cue import FaceCue

YOLO_MODEL  = "/home/a2r/yolov8n.engine"
REID_MODEL  = "/home/a2r/reid.onnx"
REID_ENGINE = "/home/a2r/reid.engine"
FRAME_W, FRAME_H = 640, 480

# Fast-path appearance gate, in RAW COSINE space (what AdaptiveGallery.match
# returns) — not the remapped [0,1] space the fusion uses.
#
# Measured 2026-09-18 (X6, n=3185 genuine / 389 impostor), converted back
# from the remapped values via cos = 2*x - 1:
#     genuine  mean 0.754,  p05 0.650
#     impostor mean 0.582,  p95 0.643
#
# The previous value was 0.55, which sits BELOW the impostor mean: about
# half of all strangers passed it and were adopted at owner_score 1.0
# without the multi-cue fusion ever running. 0.66 sits just above impostor
# p95, so a stranger has to be in the top 5% of lookalikes to get through.
FASTPATH_REID_MIN = 0.66
FX, FY = 615.0, 615.0
CX_CAM, CY_CAM = FRAME_W/2, FRAME_H/2

REID_THRESHOLD  = 0.60   # ReID similarity to match owner
OWNER_THRESHOLD = 0.65
RECAPTURE_MIN_SCORE = 0.65   # combined score to confirm owner
IOU_THRESHOLD   = 0.25
LOST_LIMIT      = 60     # frames before declaring lost
REID_INTERVAL   = 20     # frames between ReID update (realtime tuning)
GATE_MAHALANOBIS = 4.0   # KF association gate (~2 sigma in 2D pixel space)


def cosine_sim(a, b):
    if a is None or b is None: return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-6))


def iou(b1, b2):
    x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1])
    x2=min(b1[2],b2[2]); y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter+1e-6)


def color_hist(frame, bbox, bins=16):
    """HSV color histogram of bbox ROI."""
    x1,y1,x2,y2 = [int(v) for v in bbox]
    x1=max(0,x1); y1=max(0,y1); x2=min(FRAME_W,x2); y2=min(FRAME_H,y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return None
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1],None,[bins,bins],[0,180,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def hist_sim(h1, h2):
    if h1 is None or h2 is None: return 0.0
    return float(cv2.compareHist(
        h1.astype(np.float32).reshape(-1,1),
        h2.astype(np.float32).reshape(-1,1),
        cv2.HISTCMP_CORREL))


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class PersonTracker:
    def __init__(self, min_detection_confidence=0.4, min_tracking_confidence=0.4):
        self.detector = YOLO(YOLO_MODEL, task="detect")

        self.reid_enabled = False
        self._reid_engine = None
        try:
            import torch, tensorrt as trt
            _TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            with open(REID_ENGINE, "rb") as _f, trt.Runtime(_TRT_LOGGER) as _rt:
                _eng = _rt.deserialize_cuda_engine(_f.read())
            self._reid_ctx    = _eng.create_execution_context()
            # Allocate GPU tensors via torch — shares PyTorch CUDA context, no pycuda conflict
            self._reid_inp_t  = torch.zeros((1,3,256,128), dtype=torch.float32, device="cuda")
            self._reid_out_t  = torch.zeros((1,512),       dtype=torch.float32, device="cuda")
            self._reid_stream = torch.cuda.current_stream()
            self._torch       = torch
            self._reid_engine = "trt_torch"
            self.reid_enabled = True
            print("[TRACKER] ReID TRT+torch GPU enabled (F30)")
        except Exception as e:
            print(f"[TRACKER] ReID TRT failed, using ONNX CPU: {e}")
            try:
                import onnxruntime as ort
                providers = ['CPUExecutionProvider']
                self.reid_sess  = ort.InferenceSession(REID_MODEL, providers=providers)
                self.reid_input = self.reid_sess.get_inputs()[0].name
                self._reid_engine = None
                self.reid_enabled = True
                print("[TRACKER] ReID ONNX CPU enabled")
            except Exception as e2:
                print(f"[TRACKER] ReID disabled: {e2}")

        # General tracking state
        self.target_locked    = False
        self.target_bbox      = None
        self.target_position  = None
        self.lock_timestamp   = None
        self.lost_count       = 0
        self.frame_count      = 0

        # Owner enrollment state
        self.owner_enrolled      = False
        self.owner_embedding     = None   # ReID embedding
        self.owner_color_hist    = None   # HSV color histogram
        self.owner_height_ratio  = None   # bbox_height / frame_height
        self.owner_embeddings    = []     # rolling buffer for averaging
        self.enroll_frames       = 0
        self.ENROLL_FRAMES_NEEDED = 20   # frames to average for enrollment

        # [P1] Motion + appearance upgrade (owner_tracking_upgrade.py)
        self.kf        = KalmanCV2D()        # constant-velocity KF on (cx,cy)
        self.gallery   = None
        self.face_cue  = FaceCue()
        self.matcher   = MultiCueMatcher(
            reid_fn=lambda f, b: self._embed(f, b),
            face_detector=lambda f, b: self.face_cue.detect(f, b),
            face_embed_fn=lambda c: self.face_cue.embed(c),
        )
        self._cur_depth = None
        self.recapture = RecaptureManager()  # coast/recapture-window state machine
        self.perf = {"yolo_ms": 0.0, "reid_ms": 0.0, "fusion_ms": 0.0}  # Phase 2 timing
        # Deployment default. Ablation conditions set this False so that every
        # frame exercises the full multi-cue fusion being evaluated.
        self.fast_path_enabled = True

        # Warm up TRT engine — eliminates cold-start NMS timeout during enrollment
        _dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            self.detector(_dummy, conf=0.35, verbose=False)
            print("[TRACKER] TRT engine warmed up")
        except Exception as _e:
            print(f"[TRACKER] Warmup failed (non-fatal): {_e}")
            print(f"[TRACKER] Warmup failed (non-fatal): {_e}")

    # ── Detection ────────────────────────────────────────────────────────
    def _detect(self, frame):
        _t0 = time.perf_counter()
        res = self.detector(frame, conf=0.35, verbose=False)[0]
        dets = []
        for b in res.boxes:
            if int(b.cls[0]) == 0:
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                dets.append([x1, y1, x2, y2])
        self.perf["yolo_ms"] = (time.perf_counter() - _t0) * 1000
        return dets

    # ── ReID ──────────────────────────────────────────────────────────────
    def _embed(self, frame, bbox):
        if not self.reid_enabled: return None
        x1,y1,x2,y2 = [int(v) for v in bbox]
        x1=max(0,x1); y1=max(0,y1); x2=min(FRAME_W,x2); y2=min(FRAME_H,y2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return None
        img = cv2.resize(roi,(128,256))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        img = (img-[0.485,0.456,0.406])/[0.229,0.224,0.225]
        img = np.transpose(img,(2,0,1))[None].astype(np.float32)
        _t0 = time.perf_counter()
        if getattr(self, "_reid_engine", None) is not None:
            # TRT+torch path (F30) — uses PyTorch CUDA context, no pycuda conflict
            self._reid_inp_t.copy_(self._torch.from_numpy(img))
            self._reid_ctx.set_input_shape("input", (1, 3, 256, 128))
            self._reid_ctx.set_tensor_address("input",  self._reid_inp_t.data_ptr())
            self._reid_ctx.set_tensor_address("output", self._reid_out_t.data_ptr())
            self._reid_ctx.execute_async_v3(stream_handle=self._reid_stream.cuda_stream)
            self._torch.cuda.synchronize()
            emb = self._reid_out_t[0].cpu().numpy()
        else:
            # ONNX CPU fallback
            emb = self.reid_sess.run(None,{self.reid_input:img})[0][0]
        self.perf["reid_ms"] = (time.perf_counter() - _t0) * 1000
        n = np.linalg.norm(emb)+1e-6
        return emb/n

    # ── Depth ─────────────────────────────────────────────────────────────
    def _get_3d(self, depth_frame, bbox):
        # Reject degenerate boxes (zero/near-zero width or height). These come
        # from a person too close or half out of frame; sampling them yields
        # garbage. Return None so the depth cue cleanly falls back.
        x1,y1,x2,y2 = [int(v) for v in bbox]
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return None
        if depth_frame is None:
            if not hasattr(self,'_dbg_depth_t') or __import__('time').time()-self._dbg_depth_t>1.0:
                self._dbg_depth_t=__import__('time').time()
                print('[DEPTH] depth_frame is None')
            return None
        dh, dw = depth_frame.shape[:2]
        x1,y1,x2,y2 = [int(v) for v in bbox]
        cx = max(0, min(dw-1, (x1+x2)//2))
        cy = max(0, min(dh-1, (y1+y2)//2))
        try:
            # Grid sampling over the trimmed bbox (Condes et al., WAF 2020,
            # sec 3.2). The previous method took a fixed 40x40 patch at the
            # bbox centre, which on a standing person is the waist and often
            # catches background between the arms — measured ROIs spanning
            # min=8858 max=18791 mm, i.e. ten metres inside one person.
            #
            # Trimming the edges removes the background halo, and a median
            # over a sparse uniform grid is far more robust to invalid pixels
            # than a dense central patch: it needs only a handful of good
            # samples anywhere on the body rather than a clean centre.
            tx1 = max(0, x1 + int(0.20 * (x2 - x1)))
            tx2 = min(dw, x2 - int(0.20 * (x2 - x1)))
            ty1 = max(0, y1 + int(0.15 * (y2 - y1)))
            ty2 = min(dh, y2 - int(0.25 * (y2 - y1)))
            if tx2 - tx1 < 4 or ty2 - ty1 < 4:
                tx1, ty1, tx2, ty2 = (max(0, x1), max(0, y1),
                                      min(dw, x2), min(dh, y2))

            gx = np.linspace(tx1, tx2 - 1, 10).astype(int)
            gy = np.linspace(ty1, ty2 - 1, 10).astype(int)
            roi = depth_frame[np.ix_(gy, gx)]
            valid = roi[(roi > 200) & (roi < 8000)]

            # Reject an outlier-dominated sample: if the spread across the
            # body exceeds ~2 m, the grid is straddling a depth discontinuity
            # (person against a far wall) and the median is meaningless.
            if len(valid) >= 4:
                _spread = float(valid.max() - valid.min()) / 1000.0
                if _spread > 2.0:
                    _med = np.median(valid)
                    valid = valid[np.abs(valid.astype(np.int32) - _med) < 700]
            if len(valid) < 3:
                if not hasattr(self,'_dbg_depth_t') or __import__('time').time()-self._dbg_depth_t>1.0:
                    self._dbg_depth_t=__import__('time').time()
                    print(f'[DEPTH] roi shape={roi.shape} dtype={roi.dtype} '
                          f'min={roi.min() if roi.size else "empty"} max={roi.max() if roi.size else "empty"} '
                          f'valid_count=0 at bbox=({x1},{y1},{x2},{y2})')
                return None
            z = np.median(valid)/1000.0
            if z<=0 or z>8.0:
                if not hasattr(self,'_dbg_z_t') or __import__('time').time()-self._dbg_z_t>1.0:
                    self._dbg_z_t=__import__('time').time()
                    print(f'[DEPTH_Z] out of range: raw_median={np.median(valid):.1f} '
                          f'z={z:.3f}m valid_count={len(valid)} '
                          f'raw_min={valid.min()} raw_max={valid.max()}')
                return None
            x = (cx-CX_CAM)*z/FX
            y = (cy-CY_CAM)*z/FY
            return (float(x),float(y),float(z))
        except Exception as e:
            if not hasattr(self,'_dbg_depth_t') or __import__('time').time()-self._dbg_depth_t>1.0:
                self._dbg_depth_t=__import__('time').time()
                print(f'[DEPTH_ERR] {e}')
            return None

    # ── Owner multi-cue match score ───────────────────────────────────────
    def _owner_score(self, frame, bbox, emb=None, other_dets=None):
        """Multi-cue fused score (ReID+color+depth+height, confidence-weighted)."""
        if not self.owner_enrolled: return 0.0, 0.0
        # F29: skip embedding when throttled (emb=False sentinel) or use cache
        bbox_key = tuple(int(v) for v in bbox)
        if emb is False:
            emb = None   # throttled frame — no embedding this cycle
        elif emb is None:
            if getattr(self, '_emb_cache_key', None) == bbox_key:
                emb = self._emb_cache_val   # reuse cached embedding
            else:
                emb = self._embed(frame, bbox)
                self._emb_cache_key = bbox_key
                self._emb_cache_val = emb
        if self.gallery is not None and emb is not None:
            reid = self.gallery.match(emb)
        elif emb is not None:
            reid = cosine_sim(self.owner_embedding, emb)
        else:
            reid = 0.0
        pos3d = self._get_3d(self._cur_depth, bbox) if self._cur_depth is not None else None
        dist  = pos3d[2] if pos3d else None
        droi = None
        if self._cur_depth is not None:
            _x1,_y1,_x2,_y2 = [int(v) for v in bbox]
            if (_x2-_x1) > 20 and (_y2-_y1) > 20:
                _ty1=_y1+int(0.15*(_y2-_y1)); _ty2=_y1+int(0.55*(_y2-_y1))
                _tx1=_x1+int(0.25*(_x2-_x1)); _tx2=_x1+int(0.75*(_x2-_x1))
                _dh,_dw = self._cur_depth.shape[:2]
                _ty1,_ty2=max(0,_ty1),min(_dh,_ty2); _tx1,_tx2=max(0,_tx1),min(_dw,_tx2)
                if _ty2>_ty1 and _tx2>_tx1:
                    droi = self._cur_depth[_ty1:_ty2,_tx1:_tx2]
        _t0 = time.perf_counter()
        # Brightness of the person crop itself, not the frame. The frame mean
        # is a poor proxy: a subject at 40 against a wall at 180 averages
        # ~150, so whole-frame low-light adaptation never fires even though
        # every appearance cue is working from an underexposed crop.
        try:
            _bx1, _by1, _bx2, _by2 = [int(v) for v in bbox]
            _bx1 = max(0, _bx1); _by1 = max(0, _by1)
            _bx2 = min(frame.shape[1], _bx2); _by2 = min(frame.shape[0], _by2)
            _crop = frame[_by1:_by2, _bx1:_bx2]
            self._bbox_mean = float(_crop.mean()) if _crop.size else -1.0
        except Exception:
            self._bbox_mean = -1.0

        r = self.matcher.score(frame, bbox, distance_m=dist, reid_emb=emb,
                               depth_roi=droi, other_dets=other_dets)
        self.perf["fusion_ms"] = (time.perf_counter() - _t0) * 1000
        self._last_multicue = r

        # ── Per-frame CSV logging (Phase 2 instrumentation) ───────────────
        if getattr(self, '_csv_enabled', False):
            import time as _t
            cues = r.get("cues", {})
            def _c(name):  # (sim, conf) or (0,0) if absent
                v = cues.get(name, (0.0, 0.0, 0.0))
                return v[0], v[1]
            s_reid,  c_reid  = _c("reid")
            s_color, c_color = _c("color")
            s_face,  c_face  = _c("face")
            s_depth, c_depth = _c("depth")
            s_height,c_height= _c("height")
            s_body,  c_body  = _c("body3d")
            decision = ("ACCEPT" if r["is_owner"] and not r["occluded"]
                        else "HOLD" if r["score"] >= 0.30
                        else "REJECT") if not r["occluded"] else "OCCLUDED"
            row = (f"{_t.time():.6f},{self.frame_count},{r['score']:.4f},"
                   f"{r['confidence']:.4f},{r['is_owner']},{r['occluded']},"
                   f"{s_reid:.4f},{c_reid:.4f},"
                   f"{s_color:.4f},{c_color:.4f},"
                   f"{s_face:.4f},{c_face:.4f},"
                   f"{s_depth:.4f},{c_depth:.4f},"
                   f"{s_height:.4f},{c_height:.4f},"
                   f"{s_body:.4f},{c_body:.4f},"
                   f"{decision},,{r['confident_cues']},{r.get('reid_raw', 0.0):.4f},"
                   f"{r.get('occ_iou', 0.0):.4f},"
                   f"{getattr(self, '_bbox_mean', -1.0):.1f}\n")
            self._csv_fh.write(row)
            self._csv_fh.flush()

        # DEBUG: print per-cue votes (throttled to 3s)
        import time as _t
        if not hasattr(self, "_dbg_t") or _t.time() - self._dbg_t > 3.0:
            self._dbg_t = _t.time()
            cues = r.get("cues", {})
            brief = " ".join(f"{k}={v[0]:.2f}" for k,v in cues.items())
            print(f"[CUE] score={r['score']:.2f} conf={r['confidence']:.2f} "
                  f"owner={r['is_owner']} occ={r['occluded']} | {brief}")
        return float(r["score"]), float(reid)

    # ── Public API ────────────────────────────────────────────────────────
    def start_enrollment(self):
        """Begin enrolling owner features over next N frames."""
        self.owner_embeddings  = []
        self.enroll_frames     = 0
        self.owner_enrolled    = False
        print("[TRACKER] Starting owner enrollment...")
        import time as _t, os
        os.makedirs(os.path.expanduser("~/tracker_logs"), exist_ok=True)
        _fname = os.path.expanduser(
            f"~/tracker_logs/session_{_t.strftime('%Y%m%d_%H%M%S')}.csv")
        self._csv_fh = open(_fname, 'w')
        self._csv_fh.write(
            "timestamp,frame_id,S_fused,confidence,is_owner,occluded,"
            "s_reid,c_reid,s_color,c_color,s_face,c_face,"
            "s_depth,c_depth,s_height,c_height,s_body3d,c_body3d,"
            "decision,gt_id,confident_cues,reid_raw,occ_iou,bbox_mean\n")
        self._csv_enabled = True
        print(f"[TRACKER] CSV logging to {_fname}")

    def enroll_frame(self, frame, bbox) -> bool:
        """Add one frame to enrollment. Returns True when enrollment complete."""
        emb = self._embed(frame, bbox)
        if emb is not None:
            self.owner_embeddings.append(emb)
        _p3 = self._get_3d(self._cur_depth, bbox) if self._cur_depth is not None else None
        dist = _p3[2] if _p3 else None
        if dist is None and not hasattr(self, "_enroll_depth_warned"):
            self._enroll_depth_warned = True
            print("[ENROLL] No depth during enrollment — depth cue will use height fallback")
        self.matcher.enroll(frame, bbox, distance_m=dist)

        # Color histogram (only save once from first frame)
        if self.owner_color_hist is None:
            self.owner_color_hist = color_hist(frame, bbox)

        # Height ratio
        h = (bbox[3]-bbox[1])/FRAME_H
        if self.owner_height_ratio is None:
            self.owner_height_ratio = h
        else:
            self.owner_height_ratio = 0.8*self.owner_height_ratio + 0.2*h

        self.enroll_frames += 1

        if self.enroll_frames >= self.ENROLL_FRAMES_NEEDED and len(self.owner_embeddings) > 0:
            # Average embeddings
            avg = np.mean(self.owner_embeddings, axis=0)
            self.owner_embedding = avg / (np.linalg.norm(avg)+1e-6)
            self.owner_enrolled  = True
            self.gallery = AdaptiveGallery(self.owner_embedding)  # anchor = enrollment [P1]
            self.matcher.finalize_enrollment()
            self.matcher.set_owned(True)
            self.kf.reset()
            print(f"[TRACKER] Owner enrolled! {len(self.owner_embeddings)} embeddings averaged")
            print(f"[TRACKER] Height ratio={self.owner_height_ratio:.3f}")
            # Enrollment quality gate. Height ratio is a proxy for enrollment
            # range, and range determines whether the anchors are usable:
            #   0.113 (~8 m) gave a ReID anchor from a crop too small to embed;
            #   0.939 (~0.5 m) gave a torso-filling crop and a height reference
            #   that scored every normally-framed person at 0.00.
            # Both produced sessions that had to be discarded. Warn loudly so
            # the operator re-enrolls rather than collecting unusable data.
            hr = self.owner_height_ratio
            if hr is not None and not (0.45 <= hr <= 0.80):
                print(f"[TRACKER] *** ENROLLMENT WARNING: height ratio {hr:.3f} "
                      f"outside usable band [0.45, 0.80].")
                if hr < 0.45:
                    print("[TRACKER] *** Too far — ReID/colour anchors will be "
                          "built from too few pixels. Move closer (~2 m) and re-enroll.")
                else:
                    print("[TRACKER] *** Too close — height reference will reject "
                          "you at normal range. Step back (~2 m) and re-enroll.")
            return True
        return False

    def reset_owner(self):
        """Release owner — go back to searching."""
        self.owner_enrolled     = False
        self.owner_embedding    = None
        self.owner_color_hist   = None
        self.owner_height_ratio = None
        self.owner_embeddings   = []
        if getattr(self, '_csv_enabled', False):
            self._csv_fh.close()
            self._csv_enabled = False
            print("[TRACKER] CSV log closed")
        self.enroll_frames      = 0
        self.kf.reset()
        self.gallery   = None
        # reuse the FaceCue built in __init__ — reloading the models here
        # cost ~1s per reset and leaked an ONNX session each time
        self.matcher   = MultiCueMatcher(
            reid_fn=lambda f, b: self._embed(f, b),
            face_detector=lambda f, b: self.face_cue.detect(f, b),
            face_embed_fn=lambda c: self.face_cue.embed(c),
        )
        self.recapture.mark_found()
        self.unlock_target()
        print("[TRACKER] Owner reset")

    def find_owner(self, frame, depth_frame=None) -> Optional[Dict]:
        self._cur_depth = depth_frame   # for multicue depth cue
        """Find enrolled owner in frame.

        [P1] upgrade over plain per-frame re-acquire:
          1. KF predicts the owner's position first.
          2. Detections are scored (appearance via AdaptiveGallery + color +
             height) AND gated against the KF prediction — a kinematically
             impossible match (maha distance too large) is vetoed even if
             appearance looks OK, e.g. a second person of similar build.
          3. If nothing matches: coast on the KF prediction for a short
             window (occlusion/miss), then attempt appearance+motion
             recapture, and only report "lost" (None) once the whole
             recapture window has elapsed.

        Returns a detection dict, or None only when truly lost.
        """
        if not self.owner_enrolled: return None

        pred_cx, pred_cy = self.kf.predict()
        # ── LOW-LIGHT ADAPTIVE THRESHOLD ──────────────────────────────
        # In poor light, reid+color lose signal and a moderate score is
        # unreliable. Measure scene brightness and raise the acceptance bar
        # so we DON'T lock a stranger on a weak dark-frame match.
        try:
            _gray = frame.mean()   # 0-255 mean brightness
        except Exception:
            _gray = 128.0
        if _gray < 40:       # very dark
            _adaptive_thresh = OWNER_THRESHOLD + 0.15
            self._lowlight = True
        elif _gray < 70:     # dim
            _adaptive_thresh = OWNER_THRESHOLD + 0.08
            self._lowlight = True
        else:                # adequate light
            _adaptive_thresh = OWNER_THRESHOLD
            self._lowlight = False
        if self._lowlight and (not hasattr(self,"_ll_t") or __import__("time").time()-self._ll_t>2.0):
            self._ll_t=__import__("time").time()
            print(f"[LOWLIGHT] brightness={_gray:.0f} -> threshold raised to {_adaptive_thresh:.2f}")

        dets = self._detect(frame)

        # FAST PATH: Check if any detection is very close to KF prediction.
        # If so, return it immediately — skip expensive ReID+multi-cue scoring.
        # This eliminates pipeline latency for normal tracking.
        #
        # EXPERIMENTAL CONTROL: measured at 65.8% of frames in a 7.8-minute
        # session (4270/6491). Those frames are decided by Kalman gating plus
        # a single ReID threshold, NOT by the six-cue confidence-gated fusion.
        # For ablations that evaluate the fusion itself, set
        # self.fast_path_enabled = False so every frame is fully verified.
        fast_det = None
        if getattr(self, "fast_path_enabled", True) and self.kf.initialized and self.matcher._currently_owned:
            # Single detection while locked = almost certainly the owner.
            # Skip scoring entirely (verify identity only periodically).
            if len(dets) == 1 and self.kf.gate(*bbox_center(dets[0])) < GATE_MAHALANOBIS * 2.0:
                _e = self._embed(frame, dets[0])
                # Require positive evidence, not merely absence of rejection.
                # The old form accepted when the embedding was None, so a
                # failed crop silently became a confident lock.
                if (_e is not None and self.gallery is not None
                        and self.gallery.match(_e) >= FASTPATH_REID_MIN):
                    self._fastpath_sim = self.gallery.match(_e)
                    fast_det = dets[0]
            else:
                for det in dets:
                    cx, cy = (det[0]+det[2])//2, (det[1]+det[3])//2
                    if self.kf.gate(cx, cy) < GATE_MAHALANOBIS * 1.2:
                        # F28: multi-person — verify appearance before fast-path accept
                        _e = self._embed(frame, det)
                        if _e is not None and self.gallery is not None and self.gallery.match(_e) < FASTPATH_REID_MIN:
                            continue   # appearance rejected — not the owner
                        self._fastpath_sim = self.gallery.match(_e) if _e is not None else 0.0
                        fast_det = det
                        break
        if fast_det is not None:
            x1,y1,x2,y2 = fast_det
            cx,cy = (x1+x2)//2,(y1+y2)//2
            self.kf.update(cx, cy)
            self.recapture.mark_found()
            self.target_bbox = fast_det
            pos3d = self._get_3d(depth_frame, fast_det)
            self.target_position = pos3d or (cx, cy)
            # Log fast-path frames so the CSV covers every frame, not just
            # those that fall through to full multi-cue scoring.
            if getattr(self, '_csv_enabled', False):
                import time as _t
                self._csv_fh.write(
                    f"{_t.time():.6f},{self.frame_count},1.0000,1.0000,True,False,"
                    f"0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,"
                    f"0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,"
                    f"FASTPATH,,0,0.0000,0.0000,-1.0\n")
                self._csv_fh.flush()
            return {
                'person_detected': True,
                'position_2d':     (cx, cy),
                'position_3d':     pos3d,
                'bbox':            (x1, y1, x2-x1, y2-y1),
                # Report the actual gallery similarity, not a hardcoded 1.0.
                # Displaying 1.0 for every fast-path frame hid an identity
                # switch behind a perfect-looking score.
                'owner_score':     float(getattr(self, '_fastpath_sim', 0.0)),
                'predicted':       False,
                'raw_bbox':        fast_det,
            }

        best_det, best_score, best_emb, best_reid = None, _adaptive_thresh, None, 0.0
        scored = []   # (det, emb, appearance_sim) — reused for recapture below

        # SAFETY: the motion gate must NEVER be fully disabled. It used to be
        # switched off entirely while coasting (occluded) - meaning during the
        # exact window you're hidden, a person standing in front of you had
        # zero positional sanity check applied. Instead we widen tolerance
        # during coasting (drifted prediction still allowed) but never remove
        # the check outright.
        gate_active = self.kf.initialized
        gate_mult = 2.5 if self.recapture.coasting else 1.0

        # Throttle CPU-bound ReID: only recompute embeddings every REID_INTERVAL
        # frames when the KF is fresh (tight prediction, low risk of drift).
        # Always run every frame while coasting/recapturing, since that's
        # exactly when ReID evidence matters most.
        # ReID at 6 ms on GPU is no longer the bottleneck it was on CPU, so
        # the interval that made sense at 40 ms is now far too coarse: the
        # cue was abstaining for 19 frames out of 20 during motion, leaving
        # colour to hold the lock alone.
        _fast_motion = False
        if self.kf.initialized:
            try:
                _vx, _vy = float(self.kf.x[2]), float(self.kf.x[3])
                _fast_motion = (_vx*_vx + _vy*_vy) > 400.0   # ~20 px/frame
            except Exception:
                pass
        _interval = 3 if _fast_motion else 6
        run_reid = (self.recapture.coasting
                    or self.frame_count % _interval == 0)

        # SAFETY: hard distance veto. A person occluding you by standing in
        # front of the camera is, by physical necessity, CLOSER than where
        # you were last tracked. Appearance cues alone are not reliable
        # enough to accept a candidate in that situation (ReID's baseline
        # similarity for ANY person sits close to the owner's own range) -
        # so this is an unconditional reject, not a weighted vote.
        last_z = (self.target_position[2]
                   if (self.target_position and len(self.target_position) == 3)
                   else None)

        for det in dets:
            emb = self._embed(frame, det) if run_reid else False  # False = skip embed
            score, reid_sim = self._owner_score(frame, det, emb=emb, other_dets=dets)

            if gate_active:
                maha = self.kf.gate(*bbox_center(det))
                if maha >= GATE_MAHALANOBIS * gate_mult:
                    # Motion-impossible ONLY vetoes weak appearance matches.
                    # A strong ReID match overrides a stale prediction.
                    if reid_sim < 0.62:
                        score = 0.0

            if last_z is not None and depth_frame is not None:
                cand_pos3d = self._get_3d(depth_frame, det)
                # Veto if candidate is notably closer (>0.6m) - someone stepping
                # Owner moving toward camera will naturally decrease distance.
                if cand_pos3d is not None and last_z > 0 and cand_pos3d[2] < last_z - 0.6:
                    score = 0.0   # gross depth jump = different person

            # ReID-priority: a strong appearance match re-locks fast, even if
            # blurred color/height drag the fused score down after fast motion.
            if reid_sim >= 0.70 and score < best_score:
                score = max(score, 0.62)   # lift blurred-but-clearly-you frames
            scored.append((det, emb, reid_sim))
            if score > best_score:
                best_score, best_det, best_emb, best_reid = score, det, emb, reid_sim

        if best_det is not None:
            # Owner found this frame — update KF + adaptive gallery [P1]
            x1,y1,x2,y2 = best_det
            cx,cy = (x1+x2)//2,(y1+y2)//2
            self.kf.update(cx, cy)
            self.recapture.mark_found()
            if self.gallery is not None and best_emb is not None:
                self.gallery.update(best_emb, best_reid, scene_unambiguous=(len(dets) == 1))
            if hasattr(self, "_last_multicue"):
                self.matcher.update_galleries(frame, best_det, self._last_multicue, best_emb)

            pos3d = self._get_3d(depth_frame, best_det)
            self.target_bbox = best_det
            self.target_position = pos3d or (cx, cy)
            return {
                'person_detected': True,
                'position_2d':     (cx, cy),
                'position_3d':     pos3d,
                'bbox':            (x1, y1, x2-x1, y2-y1),
                'owner_score':     best_score,
                'predicted':       False,
                'annotated_frame': frame,
            }

        # ── Owner not matched this frame — lost handling [P1 recapture] ──
        self.recapture.mark_lost()

        if self.recapture.coasting and self.target_bbox is not None:
            # Brief miss/occlusion: report the KF-predicted position instead
            # of dropping out immediately — lets the caller keep moving
            # smoothly through a few dropped frames.
            bw = self.target_bbox[2] - self.target_bbox[0]
            bh = self.target_bbox[3] - self.target_bbox[1]
            pred_bbox = (pred_cx-bw/2, pred_cy-bh/2, pred_cx+bw/2, pred_cy+bh/2)
            # Clamp the coasted/predicted box to actual frame bounds - the KF
            # can extrapolate past the edges (fast motion near the edge, or
            # close-range apparent speed), which previously produced an
            # out-of-frame bbox and a silently-empty depth ROI downstream.
            pred_bbox = (
                max(0.0, min(float(FRAME_W), pred_bbox[0])),
                max(0.0, min(float(FRAME_H), pred_bbox[1])),
                max(0.0, min(float(FRAME_W), pred_bbox[2])),
                max(0.0, min(float(FRAME_H), pred_bbox[3])),
            )
            pos3d = self._get_3d(depth_frame, pred_bbox)
            return {
                'person_detected': True,
                'position_2d':     (pred_cx, pred_cy),
                'position_3d':     pos3d,
                'bbox':            (int(pred_bbox[0]), int(pred_bbox[1]), int(bw), int(bh)),
                'owner_score':     0.0,
                'predicted':       True,
                'annotated_frame': frame,
            }

        if not self.recapture.expired:
            # Try to re-lock, but IDENTITY MUST be confirmed by the full
            # multi-cue matcher - not just reid+motion. A stranger standing
            # where the owner was passes motion, so we require is_owner=True
            # AND a strong fused score to prevent owner-switching.
            cands = [(reid_sim, *bbox_center(det)) for (det, emb, reid_sim) in scored]
            i = self.recapture.try_recapture(cands, pred_cx, pred_cy)
            if i is not None:
                det = scored[i][0]
                # Re-verify identity with the FULL multi-cue matcher
                fused, _ = self._owner_score(frame, det, emb=scored[i][1], other_dets=dets)
                mc = getattr(self, "_last_multicue", {})
                confirmed = mc.get("is_owner", False) and fused >= RECAPTURE_MIN_SCORE
                if confirmed:
                    x1,y1,x2,y2 = det
                    cx,cy = (x1+x2)//2, (y1+y2)//2
                    self.kf.init(cx, cy)
                    self.recapture.mark_found()
                    pos3d = self._get_3d(depth_frame, det)
                    self.target_bbox = det
                    self.target_position = pos3d or (cx, cy)
                    return {
                        'person_detected': True,
                        'position_2d':     (cx, cy),
                        'position_3d':     pos3d,
                        'bbox':            (x1, y1, x2-x1, y2-y1),
                        'owner_score':     fused,
                        'predicted':       False,
                        'annotated_frame': frame,
                    }
                # Candidate failed identity check - do NOT lock onto a stranger

        # Truly lost — recapture window fully elapsed
        return None

    def detect_person_at_point(self, frame, nx, ny, depth_frame=None) -> Optional[Dict]:
        """Pick the detected person under a normalised image point.

        Replaces gesture-based enrollment. The operator clicks the person to
        follow, which removes a whole class of failure: wrong-person selection
        when boxes overlap, the scale-dependent finger-counting thresholds
        that failed beyond ~2 m, and the shared-timer bug when two hands were
        visible. Selection is now deterministic and independent of range.

        nx, ny are in [0,1] relative to the frame, so the client does not need
        to know the capture resolution.
        """
        h, w = frame.shape[:2]
        px = int(max(0.0, min(1.0, float(nx))) * w)
        py = int(max(0.0, min(1.0, float(ny))) * h)

        dets = self._detect(frame)
        if not dets:
            return None

        # Prefer a box containing the click. Where boxes overlap, the smallest
        # containing box is the better guess: a click inside two boxes is far
        # more likely to be aimed at the nearer, tighter one than at a large
        # box that happens to span the point.
        containing = [d for d in dets if d[0] <= px <= d[2] and d[1] <= py <= d[3]]
        if containing:
            best_det = min(containing, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        else:
            # Missed everyone — take the nearest box centre, but only if the
            # click was close. A click far from any person should fail loudly
            # rather than enrolling somebody at random.
            best_det = min(dets, key=lambda b: ((b[0]+b[2])//2 - px)**2
                                             + ((b[1]+b[3])//2 - py)**2)
            bx, by = (best_det[0]+best_det[2])//2, (best_det[1]+best_det[3])//2
            if ((bx-px)**2 + (by-py)**2) ** 0.5 > 0.25 * w:
                return None

        x1, y1, x2, y2 = best_det
        cx, cy = (x1+x2)//2, (y1+y2)//2
        return {
            'person_detected': True,
            'position_2d':     (cx, cy),
            'position_3d':     self._get_3d(depth_frame, best_det),
            'bbox':            (x1, y1, x2-x1, y2-y1),
            'annotated_frame': frame,
            'raw_bbox':        best_det,
        }

    def enroll_at_point(self, frame, nx: float, ny: float, depth_frame=None):
        """Pick the detection containing a clicked point.

        Replaces gesture enrollment. The wave gesture had three independent
        failure modes — scale-dependent finger counting, the shared timer
        across multiple hands, and largest-box selection adopting a bystander
        whose box overlapped the waver's. A click is unambiguous: the operator
        states which person, and the only ambiguity left is overlapping boxes,
        resolved by taking the smallest containing box (the innermost person)
        rather than the largest.

        nx, ny are normalised to [0,1] so the caller does not need to know the
        frame size.
        """
        dets = self._detect(frame)
        if not dets:
            return None

        h, w = frame.shape[:2]
        px, py = int(nx * w), int(ny * h)

        containing = [d for d in dets
                      if d[0] <= px <= d[2] and d[1] <= py <= d[3]]
        if containing:
            # Smallest containing box: when two people overlap, the click
            # landed inside both, but the tighter box is the person actually
            # under the cursor.
            best = min(containing, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        else:
            # Click missed everyone — take the nearest box centre, but only
            # if it is close enough that the intent is clear.
            best = min(dets, key=lambda b: ((b[0]+b[2])//2 - px)**2
                                         + ((b[1]+b[3])//2 - py)**2)
            bw = best[2] - best[0]
            if abs((best[0]+best[2])//2 - px) > bw * 1.5:
                return None

        x1, y1, x2, y2 = best
        return {
            'person_detected': True,
            'position_2d':     ((x1+x2)//2, (y1+y2)//2),
            'position_3d':     self._get_3d(depth_frame, best),
            'bbox':            (x1, y1, x2-x1, y2-y1),
            'annotated_frame': frame,
            'raw_bbox':        best,
        }

    def detect_waving_person(self, frame, wave_bbox, depth_frame=None) -> Optional[Dict]:
        """Find the person whose body contains the waving hand."""
        dets = self._detect(frame)
        if not dets: return None

        # Find person bbox that contains the waving hand
        wx,wy,ww,wh = wave_bbox
        hand_cx = wx + ww//2
        hand_cy = wy + wh//2

        # Among the boxes containing the hand, pick the one the hand most
        # plausibly BELONGS to — not the largest.
        #
        # Taking the largest was wrong: when a bystander's box overlaps the
        # waver's, the bystander is often bigger (closer to the camera), so
        # the raised hand enrolled the wrong person. A raised hand is near
        # the top of its owner's box and close to their horizontal centre,
        # so score candidates on that instead.
        best_det = None
        best_score = -1.0
        for det in dets:
            x1,y1,x2,y2 = det
            if not (x1 <= hand_cx <= x2 and y1 <= hand_cy <= y2):
                continue
            w = max(1, x2 - x1); h = max(1, y2 - y1)

            # Where the hand sits vertically: 0 at the top of the box, 1 at
            # the bottom. A raised hand should be in the upper third.
            v = (hand_cy - y1) / h
            vert = max(0.0, 1.0 - v / 0.45)        # 1.0 at the very top

            # Horizontal offset from the body centre, in box widths.
            hx = abs(hand_cx - (x1 + x2) / 2) / w
            horiz = max(0.0, 1.0 - hx / 0.6)

            score = 0.65 * vert + 0.35 * horiz
            if score > best_score:
                best_score = score
                best_det = det

        # Every containing box had the hand low or far off-centre, which
        # means it is probably not that person's hand. Fall through to the
        # proximity fallback rather than enrolling on a bad match.
        if best_score < 0.25:
            best_det = None

        # Fallback: person whose centre is nearest the hand centroid
        # (replaces largest-in-frame, which picks background subjects)
        if best_det is None:
            best_det = min(dets, key=lambda b: (
                ((b[0]+b[2])//2 - hand_cx)**2 + ((b[1]+b[3])//2 - hand_cy)**2))

        x1,y1,x2,y2 = best_det
        cx,cy = (x1+x2)//2,(y1+y2)//2
        pos3d = self._get_3d(depth_frame, best_det)

        return {
            'person_detected': True,
            'position_2d':     (cx,cy),
            'position_3d':     pos3d,
            'bbox':            (x1,y1,x2-x1,y2-y1),
            'annotated_frame': frame,
            'raw_bbox':        best_det,
        }

    def detect_person(self, frame, depth_frame=None) -> Dict:
        dets = self._detect(frame)
        result = {'person_detected':False,'position_2d':None,
                  'position_3d':None,'bbox':None,'annotated_frame':frame.copy()}
        if not dets: return result
        best = max(dets, key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
        x1,y1,x2,y2 = best
        cx,cy = (x1+x2)//2,(y1+y2)//2
        result.update({'person_detected':True,'position_2d':(cx,cy),
                       'position_3d':self._get_3d(depth_frame,best),
                       'bbox':(x1,y1,x2-x1,y2-y1)})
        return result

    def lock_target(self, person_data: Dict):
        if not person_data.get('person_detected'): return
        x,y,w,h = person_data['bbox']
        self.target_bbox     = [x,y,x+w,y+h]
        self.target_position = person_data.get('position_3d') or person_data['position_2d']
        self.target_locked   = True
        self.lock_timestamp  = time.time()
        self.lost_count      = 0
        self.frame_count     = 0
        self._bound_frames   = 0
        if self.owner_enrolled:
            cx, cy = x + w/2.0, y + h/2.0
            self.kf.init(cx, cy)
            self.recapture.mark_found()

    def unlock_target(self):
        self.target_locked   = False
        self.target_bbox     = None
        self.target_position = None
        self.lock_timestamp  = None
        self.matcher.set_owned(False)   # F31: exit permissive HOLD regime on loss

    def is_target_locked(self): return self.target_locked
    def get_target_position(self): return self.target_position

    # Geometric binding parameters. Tight by design: this path bypasses
    # identity verification entirely, so it must only fire when the
    # association is beyond reasonable doubt.
    BIND_IOU_MIN    = 0.45   # overlap with last known box
    BIND_MARGIN     = 0.20   # winner must beat runner-up by this much
    BIND_MAX_FRAMES = 45     # force re-verification periodically regardless

    def _track_bound_match(self, frame, depth_frame=None):
        """Associate the bound target geometrically, or return None.

        Returns None whenever the association is in any way ambiguous, which
        hands the decision back to the full multi-cue fusion. False negatives
        here are cheap (one frame of verification); false positives are not
        (following the wrong person).
        """
        if self.target_bbox is None:
            return None

        # Periodic re-verification. Pure geometric tracking will eventually
        # drift onto whoever happens to overlap, so identity is re-checked
        # every BIND_MAX_FRAMES even when association looks perfect.
        if getattr(self, "_bound_frames", 0) >= self.BIND_MAX_FRAMES:
            return None

        dets = self._detect(frame)
        if not dets:
            return None

        scored = sorted(((iou(self.target_bbox, d), d) for d in dets),
                        key=lambda t: t[0], reverse=True)
        best_s, best_d = scored[0]
        if best_s < self.BIND_IOU_MIN:
            return None

        # A rival with comparable overlap means two people are crossing.
        # That is exactly when misidentification happens, so verify instead.
        if len(scored) > 1 and (best_s - scored[1][0]) < self.BIND_MARGIN:
            return None

        # Motion plausibility: the Kalman gate must also agree.
        if self.kf.initialized:
            cx, cy = bbox_center(best_d)
            if self.kf.gate(cx, cy) > GATE_MAHALANOBIS:
                return None

        x1, y1, x2, y2 = best_d
        cx, cy = (x1+x2)//2, (y1+y2)//2
        self.target_bbox = best_d
        pos3d = self._get_3d(depth_frame, best_d)
        self.target_position = pos3d or (cx, cy)
        if self.kf.initialized:
            self.kf.update(cx, cy)
        self.recapture.mark_found()

        return {'person_detected': True, 'position_2d': (cx, cy),
                'position_3d': pos3d, 'bbox': (x1, y1, x2-x1, y2-y1),
                'annotated_frame': frame, 'owner_score': -1.0,
                'bound': True, 'predicted': False}

    def update_target(self, frame, depth_frame=None) -> Optional[Dict]:
        """Track target — uses owner matching if enrolled, else IoU."""
        if not self.target_locked: return None
        self.frame_count += 1
        self._emb_cache_key = None   # F29: invalidate per-frame embedding cache

        if self.owner_enrolled:
            # ── Track-bound identity (Srouji et al., arXiv:2309.12479) ──────
            #
            # While the operator-selected person remains continuously
            # associated frame to frame, identity is SETTLED and the fusion is
            # not consulted. Re-identification runs only when the binding
            # breaks.
            #
            # This is the structural fix for mid-session switching. Running
            # find_owner every frame gives the system an opportunity to change
            # its mind on every frame, and with barely-separated cue
            # distributions it eventually takes one. Srouji call re-id "only
            # when necessary" for the same reason; Condes likewise continue on
            # the established identity once confirmed: "the linear and angular
            # errors are computed, even if the face of the person is not seen
            # anymore, as the system has checked previously that that is the
            # right person."
            #
            # The binding is deliberately conservative: geometric association
            # must be unambiguous (one clear IoU winner, motion-plausible, no
            # rival close behind). Any ambiguity falls through to full
            # verification.
            bound = self._track_bound_match(frame, depth_frame)
            if bound is not None:
                self.lost_count = 0
                self._bound_frames = getattr(self, "_bound_frames", 0) + 1
                return bound

            # Binding broken — now identity must be re-established.
            self._bound_frames = 0
            pd = self.find_owner(frame, depth_frame)
            if pd:
                self.lost_count = 0
                return pd
            else:
                self.lost_count += 1
                if self.lost_count > LOST_LIMIT:
                    print("[TRACKER] Owner lost for too long — still enrolled, searching")
                    self.unlock_target()
                return None

        # Fallback IoU tracking
        dets = self._detect(frame)
        if not dets:
            self.lost_count += 1
            if self.lost_count > LOST_LIMIT:
                self.unlock_target()
            return None

        best_det = None
        best_iou = IOU_THRESHOLD
        for det in dets:
            s = iou(self.target_bbox, det) if self.target_bbox else 0
            if s > best_iou:
                best_iou = s; best_det = det

        if best_det is None:
            self.lost_count += 1
            if self.lost_count > LOST_LIMIT: self.unlock_target()
            return None

        self.lost_count  = 0
        self.target_bbox = best_det
        x1,y1,x2,y2     = best_det
        cx,cy            = (x1+x2)//2,(y1+y2)//2
        pos3d            = self._get_3d(depth_frame, best_det)
        self.target_position = pos3d or (cx,cy)

        return {'person_detected':True,'position_2d':(cx,cy),
                'position_3d':pos3d,'bbox':(x1,y1,x2-x1,y2-y1),
                'annotated_frame':frame}

    def draw_tracking_info(self, frame, person_data=None):
        ann = frame.copy()
        if self.target_locked and person_data:
            if person_data.get('bbox'):
                x,y,w,h = person_data['bbox']
                predicted = person_data.get('predicted', False)
                if predicted:
                    color = (0, 165, 255)   # orange = coasting on KF prediction
                else:
                    color = (0,255,0) if self.owner_enrolled else (0,255,255)
                cv2.rectangle(ann,(x,y),(x+w,y+h),color,3)
                if predicted:
                    label = "OWNER (predicted)"
                elif self.owner_enrolled:
                    label = f"OWNER {person_data.get('owner_score',0):.2f}"
                else:
                    label = "TARGET"
                cv2.putText(ann,label,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
            if person_data.get('position_3d'):
                px,py,pz = person_data['position_3d']
                cv2.putText(ann,f"dist={pz:.2f}m  x={px:.2f}m",
                    (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            if self.lock_timestamp:
                dur = time.time()-self.lock_timestamp
                cv2.putText(ann,f"tracking {dur:.1f}s",
                    (10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,0),2)
        return ann
