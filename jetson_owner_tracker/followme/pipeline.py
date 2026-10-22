"""FollowMe pipeline: perception (Fig. 1) + decision making + navigation.

``FollowMePerception.process`` runs, per frame:

    (b) instance segmentation -> background-removed crops
    (c) Re-ID features -> Eq. 1 distances -> target = argmin if <= lambda_d
        (during calibration: features are collected instead)
    (d) masked depth -> point cloud centroid c_p -> Kalman filter (t_exp)
    (e) gestures on the target's crop -> SVM -> xi filter

``FollowMeController.step`` turns that into alpha / beta / gamma, runs the
state machine and executes the action on a navigation backend.
"""

from __future__ import annotations

import logging
import math
import time
import csv, os  # a2r_cuelog
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from followme.calibration import CalibrationCollector, IdentityMemory, TargetModel, _unit, identify
from followme.config import FollowMeConfig
from followme.gestures import GestureRecognizer, GestureResult
from followme.localization import (CameraIntrinsics, Goal2D, TargetKalmanFilter, compute_goal,
                                   heading_angle, position_from_bbox, safety_radius, target_position,
                                   transform_point)
from followme.navigation import NavigationBackend, NavigationExecutor
from followme.reid import FeatureExtractor
from followme.segmentation import PersonInstance, PersonSegmenter, masked_crop
from followme.state_machine import Action, ActionKind, FollowMeStateMachine, Inputs, State

LOGGER = logging.getLogger(__name__)

#: camera optical frame (x right, y down, z fwd) -> robot base (x fwd, y left, z up),
#: camera at the origin. Used when no map transform is available.
OPTICAL_TO_BASE = np.array([[0, 0, 1, 0],
                            [-1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 0, 1]], dtype=np.float64)


class Mode(str, Enum):
    UNCALIBRATED = "UNCALIBRATED"
    CALIBRATING = "CALIBRATING"
    IDENTIFYING = "IDENTIFYING"


@dataclass
class PerceptionResult:
    timestamp: float
    mode: Mode
    persons: list[PersonInstance]
    distances: np.ndarray                   # Eq. 1 per person (inf = invalid / no model)
    threshold: float                        # lambda_d (nan when uncalibrated)
    target_index: Optional[int]
    p_cam: Optional[np.ndarray]             # raw centroid c_p (this frame)
    p_filt: Optional[np.ndarray]            # KF estimate c_p_filt (None when expired)
    theta: Optional[float]
    gesture: Optional[GestureResult]
    calibration_progress: float = 0.0
    identity_info: str = ""
    features: Optional[np.ndarray] = None   # (N, D) Re-ID features per person
    valid: Optional[np.ndarray] = None
    obstacles: list = field(default_factory=list)   # ObstacleInstance (chairs ...) this frame
    timings_ms: dict = field(default_factory=dict)

    @property
    def alpha(self) -> bool:
        return self.p_filt is not None


class A2RGuard:
    """a2r_idguard: keeps the owner's identity through crossings, re-finds a far owner, uses the
    phone / lidar position as a second opinion and removes the robot's own turning from the
    camera-frame Kalman filter. Every shortcut past strict Re-ID goes through _a2r_accept()."""

    def __init__(self) -> None:
        self.cross_until = -math.inf     # someone touched the owner's box: no shortcuts until then
        self.cross_t0: Optional[float] = None
        self.cross_hold_s = 1.5          # ... for this long after the last touch
        self.cross_max_s = 4.0           # a crossing never blocks the shortcuts longer than this
        self.hint = None                 # (xyz camera frame, radius m, time, source) phone or lidar
        self.hint_age_s = 0.6
        self.far_start_m = 3.0           # beyond this range Re-ID distances grow (small crops)
        self.far_gain = 0.08             # threshold x (1 + gain * (range - far_start)) ...
        self.far_max = 1.30              # ... at most this much
        self.far_confirm = 3             # far re-acquisition: frames in a row at the same spot
        self.far_margin = 0.04           # runner-up must look clearly worse
        self.depth_jump_m = 0.7          # shortcut point this much nearer/further than the track
        self.n = dict(strict=0, continuity=0, relaxed=0, far=0, crossing=0, hint_block=0,
                      depth_block=0, ego=0)
        # a2r_owner_memory: keep learning the owner's look while locked
        self.learn_s = 120.0             # first 2 min after the lock: a new view every learn_dt
        self.learn_dt, self.learn_dt_after = 0.25, 1.0
        self.learn_max = 2.0             # learn a frame up to this x threshold - only when the identity
                                         # is certain from tracking (no one else near, lidar agrees)
        self.max_views = 400
        self.lidar_factor = 2.0          # the lidar keeper vouches: accept up to this x threshold
        self.lock_t, self.learn_t = None, -math.inf
        self.n.update(lidar=0, learned=0)


class FollowMePerception:
    def __init__(self, cfg: FollowMeConfig, segmenter: PersonSegmenter, extractor: FeatureExtractor,
                 intrinsics: Optional[CameraIntrinsics] = None,
                 gestures: Optional[GestureRecognizer] = None,
                 target_model: Optional[TargetModel] = None) -> None:
        self.cfg = cfg
        self.segmenter = segmenter
        self.extractor = extractor
        self.intrinsics = intrinsics
        self.gestures = gestures
        self.kf = TargetKalmanFilter(cfg.localization)
        self._pend_n, self._pend_p = 0, None
        self.rejected_gate = 0
        self.relaxed_accepts = 0
        self.a2r = A2RGuard()                # a2r_idguard
        self.force_features = False     # set by ownership while CLAIMING
        self.memory = IdentityMemory(cfg.identity)
        self.model = target_model
        self.collector: Optional[CalibrationCollector] = None
        self.mode = Mode.IDENTIFYING if target_model is not None else Mode.UNCALIBRATED
        self.rng = np.random.default_rng(cfg.calibration.shuffle_seed)

    # -- calibration control -------------------------------------------------
    def start_calibration(self) -> None:
        self.collector = CalibrationCollector(self.cfg.calibration)
        self.mode = Mode.CALIBRATING
        self.kf.reset()
        self.memory.reset()
        LOGGER.info("Calibration started: the target should be alone in view and walk "
                    "at different distances for a few seconds")

    def finish_calibration(self) -> bool:
        """End calibration now with the samples collected so far (needs >= 6)."""
        if self.collector is None or len(self.collector.features) < 6:
            return False
        self.set_target_model(self.collector.finish(self.rng, self.cfg.identity.top_k))
        return True

    def set_target_model(self, model: TargetModel) -> None:
        self.model, self.collector = model, None
        if getattr(self, "a2r", None) is not None:
            self.a2r.lock_t = None                                 # a2r_owner_memory
        self.mode = Mode.IDENTIFYING
        self.kf.reset()
        self.memory.reset()

    def clear_target(self) -> None:
        self.model, self.collector = None, None
        self.mode = Mode.UNCALIBRATED
        self.kf.reset()
        self.memory.reset()

    # -- per frame -------------------------------------------------------------
    def process(self, frame: np.ndarray, depth_m: Optional[np.ndarray] = None,
                now: Optional[float] = None) -> PerceptionResult:
        now = time.monotonic() if now is None else now
        self._cuelog_pre(now)   # a2r_cuelog
        t0 = time.perf_counter()
        persons = self.segmenter.segment(frame)
        t1 = time.perf_counter()
        crops, valid_idx = [], []
        for i, p in enumerate(persons):
            c = masked_crop(frame, p, self.cfg.segmentation.remove_background)
            if c is not None:
                crops.append(c)
                valid_idx.append(i)
        feats = np.zeros((len(persons), self.extractor.feature_dim), np.float32)
        valid = np.zeros(len(persons), bool)
        need_features = self.mode != Mode.UNCALIBRATED or self.force_features
        if crops and need_features:
            feats[valid_idx] = self.extractor.extract(crops)
            valid[valid_idx] = True
        t2 = time.perf_counter()

        distances = np.full(len(persons), np.inf)
        target: Optional[int] = None
        how, crossing = "", False           # a2r_idguard: how the owner was accepted
        if self.mode == Mode.CALIBRATING and self.collector is not None:
            dists_m = [self._planar_distance(depth_m, persons[i]) for i in valid_idx[:1]]
            self.collector.offer(feats[valid], dists_m, now)
            if self.collector.done:
                self.set_target_model(self.collector.finish(self.rng, self.cfg.identity.top_k))
        elif self.mode == Mode.IDENTIFYING and self.model is not None and len(persons):
            ident = identify(self.model, feats, valid, self.memory, self.cfg.identity.negative_margin)
            distances, target = ident.distances, ident.target_index
            how = "strict" if target is not None else ""          # a2r_idguard
            crossing = self._a2r_crossing(persons, now)
            if target is None and not crossing:
                target = self._continuity(persons, valid, feats, distances, now)
                how = "continuity" if target is not None else ""
        t3 = time.perf_counter()

        if target is None and self.mode == Mode.IDENTIFYING and self.model is not None and len(persons) \
                and not crossing:                                   # a2r_idguard
            target = self._relaxed_candidate(depth_m, persons, feats, valid, distances, now)
            how = "relaxed" if target is not None else how
            if target is None:
                target = self._a2r_far(depth_m, persons, feats, valid, distances, now)
                how = "far" if target is not None else how
            if target is None:                                    # a2r_owner_memory
                target = self._a2r_lidar(depth_m, persons, feats, valid, distances, now)
                how = "lidar" if target is not None else how
        p_cam = None
        loc = self.cfg.localization
        if target is not None:
            cand, target = target, None
            if self.intrinsics is not None:
                if depth_m is not None:
                    p_cam = target_position(depth_m, persons[cand].mask, self.intrinsics, loc)
                elif loc.bbox_range_fallback:
                    p_cam = position_from_bbox(persons[cand].bbox, self.intrinsics, loc.assumed_person_height_m,
                                               loc.mirror_x)
            a2r_ok = self._a2r_accept(how, p_cam, now, crossing)   # a2r_idguard
            pred = self.kf.estimate(now)
            if pred is not None:
                # tracked: accept only near the prediction (gate grows while unseen)
                gate = None if loc.gate_m is None else \
                    loc.gate_m + loc.max_person_speed_mps * max(0.0, now - self.kf.t_measure)
                if not a2r_ok:
                    pass
                elif gate is None or p_cam is None or float(np.linalg.norm(p_cam - pred)) <= gate:
                    target = cand
                else:
                    self.rejected_gate += 1
                self._pend_n, self._pend_p = 0, None
            else:
                # not tracked: require confirm_frames consecutive, spatially consistent hits
                same = (self._pend_p is None or p_cam is None
                        or float(np.linalg.norm(p_cam - self._pend_p)) <= loc.confirm_radius_m)
                self._pend_n = self._pend_n + 1 if (same and self._pend_n > 0) else 1
                self._pend_p = p_cam
                need = max(loc.confirm_frames, self.a2r.far_confirm if how == "far" else 1)  # a2r_idguard
                if not a2r_ok:
                    self._pend_n, self._pend_p = 0, None
                elif self._pend_n >= need:
                    target = cand
                    self._pend_n, self._pend_p = 0, None
        else:
            self._pend_n, self._pend_p = 0, None
        if target is not None:
            self._owner_box = (persons[target].bbox.copy(), now)
        if target is None:
            p_cam = None
        else:
            if p_cam is not None:
                self.kf.update(p_cam, now)
            self.a2r.n[how or "strict"] = self.a2r.n.get(how or "strict", 0) + 1   # a2r_idguard
            self._a2r_learn(feats, persons, valid, target, distances, p_cam, how, crossing, now)  # a2r_owner_memory
            if self.model is not None and self.model.active_metric == "gallery" and \
                    (how == "strict" or (how in ("continuity", "relaxed") and not crossing)):
                self.memory.observe(self.model, feats, [p.bbox for p in persons], valid, target,
                                    float(distances[target]), now)
        p_filt = self.kf.estimate(now)
        theta = heading_angle(p_filt) if p_filt is not None else None
        t4 = time.perf_counter()

        gesture = None
        if self.gestures is not None and self.mode == Mode.IDENTIFYING:
            gesture = self.gestures.update(frame, persons[target].bbox if target is not None else None)
        t5 = time.perf_counter()
        self._cuelog(now, depth_m, persons, feats, valid, distances, target, how, crossing,  # a2r_cuelog
                     {"segment": 1e3 * (t1 - t0), "reid": 1e3 * (t2 - t1), "identify": 1e3 * (t3 - t2),
                      "localize": 1e3 * (t4 - t3), "gesture": 1e3 * (t5 - t4), "total": 1e3 * (t5 - t0)})
        return PerceptionResult(
            timestamp=now, mode=self.mode, persons=persons, distances=distances, features=feats, valid=valid,
            threshold=self.model.active_threshold if self.model is not None else float("nan"),
            target_index=target, p_cam=p_cam, p_filt=p_filt, theta=theta, gesture=gesture,
            obstacles=list(getattr(self.segmenter, "last_obstacles", [])),
            calibration_progress=self.collector.progress if self.collector is not None else 0.0,
            identity_info=(f"views +{self.memory.counters['views_added']}  strangers {len(self.memory.negatives)}  "
                           f"blocked {self.memory.counters['rejected_by_negatives']}  "
                           f"held {self.relaxed_accepts}"
                           if self.model is not None and self.model.active_metric == 'gallery' else ""),
            timings_ms={"segment": 1e3 * (t1 - t0), "reid": 1e3 * (t2 - t1), "identify": 1e3 * (t3 - t2),
                        "localize": 1e3 * (t4 - t3), "gesture": 1e3 * (t5 - t4), "total": 1e3 * (t5 - t0)})

    # -- a2r_cuelog: per-candidate log of every quantity the identity cascade uses -----------
    _CUELOG_COLS = ["t_wall", "t_mono", "frame", "mode", "n_persons", "cand", "valid", "det_score",
                    "x1", "y1", "x2", "y2", "mask_px", "d_reid", "thr", "neg_sim", "iou_prev",
                    "prev_age_s", "range_m", "qx", "qy", "qz", "kf_dist", "kf_gate", "kf_age_s",
                    "kf_pos_sd", "hint_dist", "hint_r", "hint_src", "hint_age_s", "crossing", "how",
                    "target", "chosen", "t_segment", "t_reid", "t_identify", "t_localize",
                    "t_gesture", "t_total", "gt"]

    def _cuelog_pre(self, now):
        """Snapshot, BEFORE this frame's decision, of the state the cascade compares against.
        Reads only; never advances the Kalman filter."""
        self._cl_pre = None
        try:
            if os.environ.get("A2R_CUELOG", "0") != "1":
                return
            kf, loc = self.kf, self.cfg.localization
            pred, gate, age, sd = None, float("nan"), float("nan"), float("nan")
            if kf.x is not None and not kf.expired(now):
                pred = kf.x[:3] + kf.x[3:] * max(0.0, now - kf.t_state)
                age = now - kf.t_measure
                P = getattr(kf, "P", None)
                if P is not None:
                    sd = float(np.sqrt(max(0.0, float(np.trace(P[:3, :3])) / 3.0)))
                if loc.gate_m is not None:
                    gate = loc.gate_m + loc.max_person_speed_mps * max(0.0, now - kf.t_measure)
            prev = getattr(self, "_owner_box", None)
            prev = (prev[0].copy(), prev[1]) if prev is not None else None
            self._cl_pre = (prev, pred, gate, age, sd, self._a2r_hint(now))
        except Exception:                                   # logging must never stop the robot
            self._cl_pre = None

    def _cuelog(self, now, depth_m, persons, feats, valid, distances, target, how, crossing, tms):
        pre = getattr(self, "_cl_pre", None)
        if pre is None:
            return
        try:
            if getattr(self, "_cl_w", None) is None:
                d = os.path.expanduser("~/followme_logs")
                os.makedirs(d, exist_ok=True)
                self._cl_f = open(os.path.join(d, time.strftime("cues_%Y%m%d_%H%M%S.csv")), "w", newline="")
                self._cl_w = csv.DictWriter(self._cl_f, fieldnames=self._CUELOG_COLS, restval="")
                self._cl_w.writeheader()
                self._cl_n, self._cl_tf = 0, time.monotonic()
            self._cl_n += 1
            prev, pred, gate, age, sd, hint = pre
            from followme.calibration import _iou
            thr = float(self.model.active_threshold) if self.model is not None else float("nan")
            neg = None
            if self.model is not None and self.model.active_metric == "gallery" and len(persons):
                try:
                    neg = self.memory.negative_sims(_unit(feats))
                except Exception:
                    neg = None
            base = {"t_wall": f"{time.time():.6f}", "t_mono": f"{now:.6f}", "frame": self._cl_n,
                    "mode": getattr(self.mode, "value", str(self.mode)), "n_persons": len(persons),
                    "crossing": int(bool(crossing)), "how": how or "",
                    "target": -1 if target is None else int(target)}
            for k in ("segment", "reid", "identify", "localize", "gesture", "total"):
                base["t_" + k] = f"{tms.get(k, float('nan')):.2f}"
            if not persons:
                self._cl_w.writerow(dict(base, cand=-1))
            for i, p in enumerate(persons):
                r = dict(base)
                b = p.bbox
                r.update(cand=i, valid=int(bool(valid[i])), det_score=f"{float(p.score):.3f}",
                         x1=f"{b[0]:.1f}", y1=f"{b[1]:.1f}", x2=f"{b[2]:.1f}", y2=f"{b[3]:.1f}",
                         mask_px=int(p.mask.sum()) if p.mask is not None else "",
                         d_reid=f"{float(distances[i]):.5f}" if i < len(distances) else "",
                         thr=f"{thr:.5f}", chosen=int(target is not None and int(target) == i))
                if neg is not None:
                    r["neg_sim"] = f"{float(neg[i]):.5f}"
                if prev is not None:
                    r["iou_prev"] = f"{_iou(b, prev[0]):.4f}"
                    r["prev_age_s"] = f"{now - prev[1]:.3f}"
                q = None
                if depth_m is not None and self.intrinsics is not None and valid[i]:
                    try:
                        q = target_position(depth_m, p.mask, self.intrinsics, self.cfg.localization)
                    except Exception:
                        q = None
                if q is not None:
                    r.update(range_m=f"{math.hypot(q[0], q[2]):.3f}", qx=f"{q[0]:.3f}",
                             qy=f"{q[1]:.3f}", qz=f"{q[2]:.3f}")
                    if pred is not None:
                        r["kf_dist"] = f"{float(np.linalg.norm(q - pred)):.3f}"
                    if hint is not None:
                        r["hint_dist"] = f"{math.hypot(q[0] - hint[0][0], q[2] - hint[0][2]):.3f}"
                if pred is not None:
                    r.update(kf_gate=f"{gate:.3f}", kf_age_s=f"{age:.3f}", kf_pos_sd=f"{sd:.3f}")
                if hint is not None:
                    r.update(hint_r=f"{float(hint[1]):.3f}", hint_src=str(hint[3]),
                             hint_age_s=f"{now - hint[2]:.3f}")
                self._cl_w.writerow(r)
            if time.monotonic() - self._cl_tf > 1.0:
                self._cl_f.flush()
                self._cl_tf = time.monotonic()
        except Exception as exc:                            # logging must never stop the robot
            if not getattr(self, "_cl_warned", False):
                self._cl_warned = True
                LOGGER.warning("a2r_cuelog disabled after error: %s", exc)
            self._cl_pre = None

    def _continuity(self, persons, valid, feats, distances, now) -> Optional[int]:
        """Tracking-by-detection: the box that clearly continues last frame's owner box (and
        overlaps nobody else's) stays the owner, if its appearance is still roughly right and it
        does not look more like a known stranger. Re-ID is then needed only to re-acquire."""
        c = self.cfg.identity
        prev = getattr(self, "_owner_box", None)
        if c.continuity_iou <= 0 or prev is None or now - prev[1] > 0.5 or self.model is None:
            return None
        from followme.calibration import _iou
        ious = np.array([_iou(p.bbox, prev[0]) if valid[i] else 0.0 for i, p in enumerate(persons)])
        if ious.size == 0:
            return None
        k = int(np.argmax(ious))
        if ious[k] < c.continuity_iou or (ious.size > 1 and np.sort(ious)[-2] > 0.1):
            return None                                  # no clear continuation / ambiguous crossing
        if not distances[k] <= c.continuity_factor * self.model.active_threshold:
            return None
        if self.model.active_metric == "gallery":
            neg = self.memory.negative_sims(_unit(feats[k:k + 1]))
            if neg is not None and (1.0 - distances[k]) - neg[0] < c.negative_margin:
                return None
        self.continuity_accepts = getattr(self, "continuity_accepts", 0) + 1
        return k

    # -- a2r_idguard ---------------------------------------------------------------------
    def a2r_ego(self, T_rel: np.ndarray) -> None:
        """The robot moved/turned: T_rel maps the previous camera frame into the current one.
        Moves the Kalman state with it, so a fast turn does not push the owner out of the gate."""
        kf = self.kf
        if kf.x is None:
            return
        R, t = T_rel[:3, :3], T_rel[:3, 3]
        kf.x[:3] = R @ kf.x[:3] + t
        kf.x[3:] = R @ kf.x[3:]
        B = np.zeros((6, 6))
        B[:3, :3] = R
        B[3:, 3:] = R
        kf.P = B @ kf.P @ B.T
        if self._pend_p is not None:
            self._pend_p = R @ self._pend_p + t
        self.a2r.n["ego"] += 1

    def _a2r_crossing(self, persons, now: float) -> bool:
        """Someone else's box reaches the owner's last box (a person walking past, through the
        line of sight or beside them). Until cross_hold_s after that, only strict Re-ID counts:
        the box-continuity and relaxed shortcuts are exactly what hands the owner to a crosser."""
        g = self.a2r
        prev = getattr(self, "_owner_box", None)
        if prev is not None and now - prev[1] <= 0.7 and len(persons) >= 2:
            b = prev[0]
            pad = 0.15 * float(b[2] - b[0])
            touch = sum(1 for p in persons if p.bbox[0] <= b[2] + pad and p.bbox[2] >= b[0] - pad)
            if touch >= 2:
                if g.cross_t0 is None or now > g.cross_until:
                    g.cross_t0 = now
                    g.n["crossing"] += 1
                g.cross_until = now + g.cross_hold_s
        return g.cross_t0 is not None and now <= g.cross_until and now - g.cross_t0 <= g.cross_max_s

    def _a2r_hint(self, now: float):
        h = self.a2r.hint
        return h if h is not None and now - h[2] <= self.a2r.hint_age_s else None

    def _a2r_accept(self, how: str, p_cam, now: float, crossing: bool) -> bool:
        """Second opinion for every acceptance that is not plain strict Re-ID (and for strict
        Re-ID during a crossing): same depth as the track, and where the phone / lidar say."""
        if p_cam is None or (how == "strict" and not crossing):
            return True
        g = self.a2r
        if self.kf.x is not None and not self.kf.expired(now):
            pred = self.kf.x[:3] + self.kf.x[3:] * max(0.0, now - self.kf.t_state)
            dt = max(0.0, now - self.kf.t_measure)
            if abs(math.hypot(p_cam[0], p_cam[2]) - math.hypot(pred[0], pred[2])) > g.depth_jump_m + 1.0 * dt:
                g.n["depth_block"] += 1
                return False
        h = self._a2r_hint(now)
        if h is not None and math.hypot(p_cam[0] - h[0][0], p_cam[2] - h[0][2]) > h[1]:
            g.n["hint_block"] += 1
            return False
        if how == "far" and h is None:
            return False                 # far re-acquisition needs the phone or the lidar to agree
        return True

    def _a2r_lidar(self, depth_m, persons, feats, valid, distances, now) -> Optional[int]:
        """a2r_owner_memory: the lidar keeper is following the owner's legs. The ONE person the camera
        sees at that spot is the owner if they look roughly right (<= lidar_factor x threshold) and
        not more like a known stranger - the camera then keeps the owner when Re-ID alone would not."""
        g, h = self.a2r, self._a2r_hint(now)
        if h is None or h[3] != "lidar" or depth_m is None or self.intrinsics is None or self.model is None:
            return None
        hits = []
        for i, p in enumerate(persons):
            if not (valid[i] and np.isfinite(distances[i])):
                continue
            q = target_position(depth_m, p.mask, self.intrinsics, self.cfg.localization)
            if q is not None and math.hypot(q[0] - h[0][0], q[2] - h[0][2]) <= h[1]:
                hits.append(i)
        if len(hits) != 1 or not distances[hits[0]] <= g.lidar_factor * self.model.active_threshold:
            return None
        k = hits[0]
        if self.model.active_metric == "gallery":
            neg = self.memory.negative_sims(_unit(feats[k:k + 1]))
            if neg is not None and (1.0 - distances[k]) - neg[0] < self.cfg.identity.negative_margin:
                return None
        return k

    def _a2r_learn(self, feats, persons, valid, target, distances, p_cam, how, crossing, now) -> None:
        """a2r_owner_memory: add the owner's current look to the gallery - 4 per second for the first
        learn_s after the lock, then 1 per second - whenever the identity is certain from tracking:
        nobody else beside the owner, no crossing, the lidar/phone agree, not more like a stranger."""
        g = self.a2r
        if self.model is None or self.model.active_metric != "gallery" or crossing or how == "far":
            return
        if g.lock_t is None:
            g.lock_t = now
        if now - g.learn_t < (g.learn_dt if now - g.lock_t <= g.learn_s else g.learn_dt_after):
            return
        thr = self.model.active_threshold
        d = float(distances[target])
        if not d <= g.learn_max * thr:
            return
        b = persons[target].bbox
        pad = 0.25 * float(b[2] - b[0])
        if sum(1 for i, p in enumerate(persons) if valid[i] and p.bbox[0] <= b[2] + pad and p.bbox[2] >= b[0] - pad) != 1:
            return
        h = self._a2r_hint(now)
        if h is not None and p_cam is not None and math.hypot(p_cam[0] - h[0][0], p_cam[2] - h[0][2]) > h[1]:
            return
        if h is None and how != "strict" and d > thr:
            return                       # nothing but the camera vouches: only learn what already matches
        mem = self.memory
        x = _unit(feats[target:target + 1])[0]
        neg = mem.negative_sims(x[None])
        if neg is not None and (1.0 - d) - neg[0] < self.cfg.identity.negative_margin:
            return
        g.learn_t = now
        existing = self.model.gallery if not mem.online else np.vstack([self.model.gallery, np.stack(mem.online)])
        if float((existing @ x).max()) >= mem.cfg.online_duplicate_sim:
            return
        mem.online.append(x)
        del mem.online[:-g.max_views]
        g.n["learned"] += 1
        mem.counters["views_added"] += 1          # shows as "views +N" on the web view

    def _a2r_far(self, depth_m, persons, feats, valid, distances, now) -> Optional[int]:
        """Far owner: crops are small and Re-ID distances grow with range. The best match may be
        up to far_max x threshold if it is clearly the best person, not more like a known
        stranger, and (checked in _a2r_accept) where the phone / lidar put the owner."""
        g = self.a2r
        if self.intrinsics is None or depth_m is None or self.model is None or self._a2r_hint(now) is None:
            return None
        ok = np.asarray(valid, bool) & np.isfinite(distances)
        if not ok.any():
            return None
        order = np.argsort(np.where(ok, distances, np.inf))
        k = int(order[0])
        if len(order) > 1 and ok[order[1]] and distances[order[1]] - distances[k] < g.far_margin:
            return None
        p = target_position(depth_m, persons[k].mask, self.intrinsics, self.cfg.localization)
        if p is None:
            return None
        scale = min(g.far_max, 1.0 + g.far_gain * max(0.0, math.hypot(p[0], p[2]) - g.far_start_m))
        if scale <= 1.0 or not distances[k] <= scale * self.model.active_threshold:
            return None
        if self.model.active_metric == "gallery":
            neg = self.memory.negative_sims(_unit(feats[k:k + 1]))
            if neg is not None and (1.0 - distances[k]) - neg[0] < self.cfg.identity.negative_margin:
                return None
        return k

    def _relaxed_candidate(self, depth_m, persons, feats, valid, distances, now) -> Optional[int]:
        """Hysteresis: while the target is tracked, accept a slightly worse match
        (<= tracking_threshold_factor * threshold) if it is where the KF expects the target
        and does not look more like a known stranger. Removes one-frame drop-outs."""
        factor = self.cfg.identity.tracking_threshold_factor
        loc = self.cfg.localization
        if factor <= 1.0 or loc.gate_m is None or self.intrinsics is None:
            return None
        pred = self.kf.estimate(now)
        if pred is None:
            return None
        ok = valid & np.isfinite(distances) & (distances <= factor * self.model.active_threshold)
        if self.model.active_metric == "gallery":
            neg = self.memory.negative_sims(_unit(feats))
            if neg is not None:
                ok &= (1.0 - distances) - neg >= self.cfg.identity.negative_margin
        gate = loc.gate_m + loc.max_person_speed_mps * max(0.0, now - self.kf.t_measure)
        for i in np.argsort(np.where(ok, distances, np.inf)):
            if not ok[i]:
                break
            if depth_m is not None:
                p = target_position(depth_m, persons[i].mask, self.intrinsics, loc)
            elif loc.bbox_range_fallback:
                p = position_from_bbox(persons[i].bbox, self.intrinsics, loc.assumed_person_height_m,
                                       loc.mirror_x)
            else:
                p = None
            if p is not None and float(np.linalg.norm(p - pred)) <= gate:
                self.relaxed_accepts += 1
                return int(i)
        return None

    def _planar_distance(self, depth_m: Optional[np.ndarray], person: PersonInstance) -> float:
        if depth_m is None or self.intrinsics is None:
            return float("nan")
        p = target_position(depth_m, person.mask, self.intrinsics, self.cfg.localization)
        return float(math.hypot(p[0], p[2])) if p is not None else float("nan")


class FollowMeController:
    """alpha/beta/gamma -> state machine -> navigation backend."""

    def __init__(self, cfg: FollowMeConfig, backend: NavigationBackend) -> None:
        self.cfg = cfg
        nav = cfg.navigation
        self.safety_radius = (nav.safety_radius_m if nav.safety_radius_m is not None
                              else safety_radius(nav.max_velocity_mps, cfg.localization.expiration_time_s,
                                                 nav.safety_factor))
        self.machine = FollowMeStateMachine(cfg.state_machine, nav.search_max_angle_rad)
        self.executor = NavigationExecutor(nav, backend)
        self.last_distance = math.inf
        self.last_alpha = False
        self.last_goal: Optional[Goal2D] = None
        LOGGER.info("Safety circle radius %.2f m", self.safety_radius)

    @property
    def state(self) -> State:
        return self.machine.state

    def step(self, result: PerceptionResult, T_map_cam: Optional[np.ndarray] = None,
             robot_yaw: Optional[float] = None) -> Action:
        goal = None
        if result.p_filt is not None:
            T = T_map_cam if T_map_cam is not None else OPTICAL_TO_BASE
            p_use = result.p_filt
            if self.cfg.navigation.steer_from_measurement and result.p_cam is not None:
                p_use = result.p_cam                                        # real-time: no filter lag
            p_map = transform_point(T, p_use)                               # Eq. 2
            goal = compute_goal(p_map, T[:3, 3], self.cfg.navigation.goal_standoff_m)
            self.last_distance = goal.distance
            self.last_goal = goal
            if result.p_cam is not None:           # real detection this frame
                self.machine.note_heading(heading_angle(result.p_cam))
        self.last_alpha = result.alpha
        beta = self.last_distance <= self.safety_radius
        gamma = result.gesture.command if result.gesture is not None else None
        action = self.machine.step(Inputs(alpha=result.alpha, beta=beta, gamma=gamma, goal=goal,
                                          search_angle_done=self.executor.search_angle))
        self.executor.execute(action, result.timestamp, robot_yaw)
        return action

    def stop(self, now: float) -> None:
        """Cancel any goal and command zero velocity (reset / shutdown)."""
        self.executor.execute(Action(ActionKind.STOP), now)
        self.executor.last_kind = None

    def command(self, gamma: str, now: float) -> Action:
        """Inject a command (keyboard / service) as if it were a gesture."""
        action = self.machine.step(Inputs(alpha=self.last_alpha, beta=self.last_distance <= self.safety_radius,
                                          gamma=gamma, goal=self.last_goal if self.last_alpha else None,
                                          search_angle_done=self.executor.search_angle))
        self.executor.execute(action, now)
        return action
