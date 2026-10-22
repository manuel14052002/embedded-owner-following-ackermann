"""Robot ownership by gesture (extension, not in the paper).

    NO_OWNER  robot stopped. The hands of the (up to) ``max_candidates`` largest people
              are checked. Exactly one person holding a RAISED OPEN PALM for
              ``claim_hold_s`` starts a claim (two at once: ignored).
    CLAIMING  the claimant is followed by box overlap only (no Re-ID model yet) while
              ~``calibration_s`` of their views are collected (frames where they overlap
              someone are skipped). Everyone else seen meanwhile becomes a known stranger.
              Claimant lost for ``claim_lost_s`` -> back to NO_OWNER.
    OWNED     normal FollowMe (fist = FOLLOW, open palm = WAIT, owner's hands only).
              Raised hands of others are ignored. The owner holding a RAISED open palm for
              ``release_hold_s`` (it first acts as WAIT: the robot stops), or the owner
              unseen for ``owner_lost_release_s`` -> NO_OWNER.

"Raised" = wrist in the top ``raised_fraction`` of the person's box and horizontally
inside it; the open palm itself is the gesture classifier's WAIT class.

:meth:`OwnershipManager.update` returns events for the caller:
``claim_started``, ``claim_aborted``, ``claimed``, ``follow``, ``released``.
"""

from __future__ import annotations

import dataclasses
import threading
import logging
from enum import Enum
from typing import Optional

import numpy as np

from followme.calibration import CalibrationCollector, _iou, _unit
from followme.config import FollowMeConfig
from followme.gestures import WAIT, landmark_features

LOGGER = logging.getLogger(__name__)


class OwnerState(str, Enum):
    NO_OWNER = "NO_OWNER"
    CLAIMING = "CLAIMING"
    OWNED = "OWNED"


def hand_is_raised(points_xy: np.ndarray, bbox: np.ndarray, fraction: float) -> bool:
    """Wrist (landmark 0) in the top ``fraction`` of the box and horizontally inside it."""
    wx, wy = float(points_xy[0, 0]), float(points_xy[0, 1])
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return x1 <= wx <= x2 and wy <= y1 + fraction * (y2 - y1)


class OwnershipManager:
    def __init__(self, cfg: FollowMeConfig, perception) -> None:
        self.cfg = cfg
        self.oc = cfg.ownership
        self.perception = perception
        self.state = OwnerState.OWNED if perception.model is not None else OwnerState.NO_OWNER
        self.message = ""
        self.claimant_box: Optional[np.ndarray] = None
        self._cand_box: Optional[np.ndarray] = None
        self._cand_start = self._cand_last = 0.0
        self._claim_seen = 0.0
        self._collector: Optional[CalibrationCollector] = None
        self._strangers: list[np.ndarray] = []
        self._owner_seen = 0.0
        self._release_start: Optional[float] = None
        self._release_last = 0.0
        self._cooldown_until = 0.0
        self._follow_until: Optional[float] = None
        self._armed_until = 0.0
        self._initialised = False
        self._rr = 0                                   # round-robin offset for hand checks
        self._frame_no = 0
        self._hand_busy = False
        self._hand_lock = threading.Lock()
        self._raised_recent: list[tuple[np.ndarray, float]] = []   # (box, time) raised palms
        self._pending_select: Optional[tuple[float, float, float]] = None   # (x, y, t)

    # -- hand readings (overridable in tests) -----------------------------------
    def _read_hand(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[tuple[str, bool]]:
        """(label, raised) for the hand inside this person's crop, or None."""
        rec = self.perception.gestures
        crop, origin = rec.target_crop(frame, bbox)
        hand = rec.detector.detect(crop)
        if hand is None:
            return None
        label = rec.classifier.predict_one(
            landmark_features(hand.landmarks, hand.handedness, rec.cfg.mirror_left_hands))
        pts = hand.landmarks[:, :2] + np.asarray(origin, np.float64)
        return label, hand_is_raised(pts, bbox, self.oc.raised_fraction)

    def _check_hands(self, frame, boxes, t) -> None:
        for bbox in boxes:
            r = self._read_hand(frame, bbox)
            if r is not None and r[0] == WAIT and r[1]:
                with self._hand_lock:
                    self._raised_recent.append((bbox, t))

    def _hand_worker(self, frame, boxes, t) -> None:
        try:
            self._check_hands(frame, boxes, t)
        finally:
            self._hand_busy = False

    def _owner_hand(self, frame: np.ndarray, result) -> Optional[tuple[str, bool]]:
        """Reuse the owner's hand already found by the gesture recognizer this frame."""
        g = result.gesture
        if g is None or g.hand is None or g.raw is None or result.target_index is None:
            return None
        pts = g.hand.landmarks[:, :2] + np.asarray(g.crop_origin, np.float64)
        return g.raw, hand_is_raised(pts, result.persons[result.target_index].bbox, self.oc.raised_fraction)

    # -- control ----------------------------------------------------------------
    def release(self, now: float, reason: str = "released") -> None:
        self.perception.clear_target()
        self.state = OwnerState.NO_OWNER
        self.claimant_box, self._cand_box, self._collector = None, None, None
        self._release_start, self._follow_until = None, None
        self._cooldown_until = now + self.oc.release_cooldown_s
        self.message = f"Robot free ({reason}) - raise an open palm to claim"
        LOGGER.info("Ownership released: %s", reason)

    def arm(self, now: float) -> bool:
        """Voice: open a short window in which raising a hand claims the robot quickly.
        Only while nobody owns the robot (an owner must release first)."""
        if self.state != OwnerState.NO_OWNER:
            return False
        self._armed_until = now + self.oc.voice_arm_s
        self._cooldown_until = 0.0
        self.message = f"Listening: raise your open palm now ({self.oc.voice_arm_s:.0f} s)"
        return True

    def select(self, x: float, y: float, now: float) -> bool:
        """Operator clicked on the video (x, y in pixels, or 0..1 normalised).
        The person under the click becomes the claimant on the next frame.
        Only while nobody owns the robot."""
        if self.state != OwnerState.NO_OWNER:
            return False
        self._pending_select = (float(x), float(y), now)
        return True

    def _take_selection(self, frame, result, now) -> Optional[np.ndarray]:
        sel, self._pending_select = self._pending_select, None
        if sel is None or now - sel[2] > 2.0 or not result.persons:
            return None
        x, y = sel[0], sel[1]
        if x <= 1.0 and y <= 1.0:                     # normalised coordinates
            x, y = x * frame.shape[1], y * frame.shape[0]
        inside = [p for p in result.persons if p.bbox[0] <= x <= p.bbox[2] and p.bbox[1] <= y <= p.bbox[3]]
        if inside:                                    # smallest box under the click (front-most)
            return min(inside, key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1])).bbox
        cx = [(0.5 * (p.bbox[0] + p.bbox[2]), 0.5 * (p.bbox[1] + p.bbox[3])) for p in result.persons]
        d = [((c[0] - x) ** 2 + (c[1] - y) ** 2) ** 0.5 for c in cx]
        k = int(np.argmin(d))
        return result.persons[k].bbox if d[k] < 0.15 * frame.shape[1] else None

    def armed(self, now: float) -> bool:
        return self.state == OwnerState.NO_OWNER and now < self._armed_until

    def _start_claim(self, box: np.ndarray, now: float) -> None:
        cal = dataclasses.replace(self.cfg.calibration, duration_s=self.oc.calibration_s,
                                  require_single_person=True)
        self._collector = CalibrationCollector(cal)
        self._strangers = []
        self.claimant_box = box.copy()
        self._claim_seen = now
        self.state = OwnerState.CLAIMING
        LOGGER.info("Claim started")

    # -- per frame ----------------------------------------------------------------
    def update(self, frame: np.ndarray, depth_m: Optional[np.ndarray], result, now: float) -> list[str]:
        if not self._initialised:
            self._owner_seen, self._initialised = now, True
            if self.state == OwnerState.NO_OWNER:
                self.message = "Robot free - raise an open palm to claim"
        # keep in sync with manual calibration (C key) / external resets
        if self.state == OwnerState.NO_OWNER and self.perception.model is not None:
            self.state, self._owner_seen = OwnerState.OWNED, now
        elif self.state == OwnerState.OWNED and self.perception.model is None:
            self.state = OwnerState.NO_OWNER
        if self.state == OwnerState.NO_OWNER:
            events = self._no_owner(frame, result, now)
        elif self.state == OwnerState.CLAIMING:
            events = self._claiming(depth_m, result, now)
        else:
            events = self._owned(frame, result, now)
        # Re-ID features are only needed while learning a claimant (and when owned)
        self.perception.force_features = self.state == OwnerState.CLAIMING
        return events

    def _no_owner(self, frame, result, now) -> list[str]:
        if self._pending_select is not None:
            box = self._take_selection(frame, result, now)
            if box is not None:
                self._armed_until = 0.0
                self._start_claim(box, now)
                self.message = "Selected! Stay in view and turn around slowly"
                return ["claim_started"]
            self.message = "Nobody at the clicked position - click on a person"
        if now < self._cooldown_until or not result.persons:
            return []
        order = list(np.argsort([-(p.bbox[3] - p.bbox[1]) for p in result.persons])[:self.oc.max_candidates])
        self._frame_no += 1
        n = self.oc.hands_per_frame
        if (self.oc.hand_check_every > 1 and self._frame_no % self.oc.hand_check_every) or \
                (self.oc.hand_async and self._hand_busy):
            checked = []                               # skip this frame (rotation paused)
        elif n and n < len(order):                     # rotate: only n MediaPipe calls this frame
            start = self._rr % len(order)
            checked = [order[(start + k) % len(order)] for k in range(n)]
            self._rr += n
        else:
            checked = order
        boxes = [result.persons[i].bbox.copy() for i in checked]
        if self.oc.hand_async:
            if boxes:
                self._hand_busy = True
                threading.Thread(target=self._hand_worker, args=(frame.copy(), boxes, now), daemon=True).start()
        else:
            self._check_hands(frame, boxes, now)
        # a raised palm seen within gesture_gap_s still counts (the person may not have been
        # checked this very frame); match it to this frame's boxes
        with self._hand_lock:
            self._raised_recent = [(b, t) for b, t in self._raised_recent if now - t <= self.oc.gesture_gap_s]
            recent = list(self._raised_recent)
        raised = [int(i) for i in order
                  if any(_iou(result.persons[i].bbox, b) >= 0.3 for b, _ in recent)]
        if len(raised) > 1:
            self._cand_box = None
            self.message = "Only ONE person may raise a hand to claim the robot"
            return []
        if len(raised) == 1:
            box = result.persons[raised[0]].bbox
            if self._cand_box is None or _iou(self._cand_box, box) < 0.3:
                self._cand_start = now
            self._cand_box, self._cand_last = box.copy(), now
            held = now - self._cand_start
            need = self.oc.voice_claim_hold_s if self.armed(now) else self.oc.claim_hold_s
            if held >= need:
                self._armed_until = 0.0
                self._start_claim(box, now)
                self.message = "Claimed! Stay in view, walk near/far and turn around"
                return ["claim_started"]
            self.message = f"Keep your hand up... {need - held:.1f} s"
            return []
        if self._cand_box is not None and now - self._cand_last > self.oc.gesture_gap_s:
            self._cand_box = None
            self.message = "Robot free - raise an open palm to claim"
        if self._armed_until and now >= self._armed_until:
            self._armed_until = 0.0
            self.message = "Robot free - raise an open palm to claim"
        return []

    def _match_claimant(self, persons, result) -> int:
        """Find the claimant this frame. Box overlap alone loses people who turn, raise
        their arms or move quickly (the box changes shape), so fall back to the nearest box
        centre, and - once some views are collected - to appearance."""
        if not persons:
            return -1
        cb = self.claimant_box
        ious = [_iou(p.bbox, cb) for p in persons]
        k = int(np.argmax(ious))
        if ious[k] >= 0.2:
            return k
        cw = cb[2] - cb[0]
        ccx, ccy = 0.5 * (cb[0] + cb[2]), 0.5 * (cb[1] + cb[3])
        dists = [np.hypot(0.5 * (p.bbox[0] + p.bbox[2]) - ccx, 0.5 * (p.bbox[1] + p.bbox[3]) - ccy)
                 for p in persons]
        near = [i for i in np.argsort(dists) if dists[i] <= max(1.0 * cw, 80.0)]
        feats = result.features
        coll = self._collector.features if self._collector is not None else []
        if near and len(near) > 1 and feats is not None and len(coll) >= 5:
            proto = _unit(np.mean(np.stack(coll), axis=0))[0]
            sims = [float(_unit(feats[i])[0] @ proto) for i in near]
            return int(near[int(np.argmax(sims))])       # two people nearby: pick by appearance
        return int(near[0]) if near else -1

    def _claiming(self, depth_m, result, now) -> list[str]:
        persons = result.persons
        k = self._match_claimant(persons, result)
        if k < 0:
            if now - self._claim_seen > self.oc.claim_lost_s:
                self.release(now, "claimant lost during calibration")
                self._cooldown_until = now
                return ["claim_aborted"]
            return []
        self._claim_seen = now
        self.claimant_box = persons[k].bbox.copy()
        valid = result.valid if result.valid is not None else np.ones(len(persons), bool)
        overlap = any(_iou(p.bbox, self.claimant_box) > 0.05 for j, p in enumerate(persons) if j != k)
        if valid[k] and not overlap:
            dist = self.perception._planar_distance(depth_m, persons[k])
            self._collector.offer(result.features[k][None], [dist], now)
        for j, p in enumerate(persons):
            if j != k and valid[j] and _iou(p.bbox, self.claimant_box) < 0.05:
                self._strangers.append(_unit(result.features[j])[0])
        del self._strangers[:-self.cfg.identity.max_negatives]
        self.message = f"Learning the new owner: {int(100 * self._collector.progress)} %  (turn around slowly)"
        if not self._collector.done:
            return []
        model = self._collector.finish(self.perception.rng, self.cfg.identity.top_k)
        self.perception.set_target_model(model)
        self.perception.memory.negatives = list(self._strangers)
        self.state = OwnerState.OWNED
        self._owner_seen = now
        self._follow_until = now + self.oc.follow_timeout_s if self.oc.start_following else None
        self.message = "New owner set" + (" - following" if self.oc.start_following else " - show a fist to follow")
        LOGGER.info("Claim complete: new owner (%d stranger views)", len(self._strangers))
        return ["claimed"]

    def _owned(self, frame, result, now) -> list[str]:
        events: list[str] = []
        if result.target_index is not None:
            self._owner_seen = now
        elif self.oc.owner_lost_release_s > 0 and now - self._owner_seen > self.oc.owner_lost_release_s:
            self.release(now, f"owner unseen {self.oc.owner_lost_release_s:.0f} s")
            return ["released"]
        if self._follow_until is not None:
            if result.alpha:
                self._follow_until = None
                events.append("follow")
            elif now > self._follow_until:
                self._follow_until = None
        r = self._owner_hand(frame, result)
        if r is not None and r[0] == WAIT and r[1]:
            if self._release_start is None:
                self._release_start = now
            self._release_last = now
            held = now - self._release_start
            if held >= self.oc.release_hold_s:
                self.release(now)
                return events + ["released"]
            self.message = f"Releasing in {self.oc.release_hold_s - held:.1f} s (keep hand up)"
        elif self._release_start is not None and now - self._release_last > self.oc.gesture_gap_s:
            self._release_start = None
            self.message = ""
        return events
