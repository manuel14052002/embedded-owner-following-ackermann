"""The identity-preserving owner tracker and its state machine.

Per frame::

    detections -> Kalman prediction -> Re-ID embeddings -> per-candidate scores
    -> hard gates -> threshold + margin rule -> temporal confirmation
    -> motion/identity update (only on ACCEPT) -> confidence -> state

State machine (driven by decisions and by the smoothed confidence)::

    TRACKING <-> LOW_CONFIDENCE <-> AMBIGUOUS
         \\            |               /
          +------> LOST (search mode) <+
                       |
                 REIDENTIFYING (candidate collecting evidence)
                       |
                   TRACKING

The owner identity (``owner_id``) is assigned once at the operator's click and
can only be cleared by an explicit :meth:`OwnerTracker.reset`.
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from owner_tracking.association import AssociationOutcome, AssociationScorer, evaluate_candidates
from owner_tracking.config import TrackerConfig
from owner_tracking.confidence import ConfidenceManager
from owner_tracking.datatypes import (
    STATE_SEVERITY, CandidateScore, ConfidenceReport, Decision, Detection, FrameResult, TrackState)
from owner_tracking.detector import PersonDetector
from owner_tracking.geometry import bbox_center, contains_point, iou, iou_matrix
from owner_tracking.identity import IdentityManager
from owner_tracking.motion import KalmanMotionPredictor, MotionPredictor, ShortTermTracklets
from owner_tracking.reid import PersonReID

LOGGER = logging.getLogger(__name__)


@dataclass
class _PendingHypothesis:
    """A candidate that must stay best for several consecutive frames."""

    bbox: np.ndarray
    count: int
    started_frame: int
    tracklet_id: int = -1


def find_clicked_detection(detections: Sequence[Detection], point: tuple[float, float],
                           ambiguity_ratio: float) -> tuple[Optional[int], str]:
    """Return ``(index, reason)`` of the detection the operator clicked.

    Among boxes containing the click, the one whose centre is closest
    (normalised by box size) wins. If two boxes are nearly equally close the
    click is rejected as ambiguous rather than guessed.
    """
    x, y = point
    hits: list[tuple[float, int]] = []
    for i, det in enumerate(detections):
        if contains_point(det.bbox, x, y):
            cx, cy = bbox_center(det.bbox)
            w = max(float(det.bbox[2] - det.bbox[0]), 1.0)
            h = max(float(det.bbox[3] - det.bbox[1]), 1.0)
            hits.append((math.hypot((x - cx) / w, (y - cy) / h), i))
    if not hits:
        return None, "click is outside every person detection"
    hits.sort()
    if len(hits) > 1 and hits[1][0] - hits[0][0] < ambiguity_ratio:
        return None, "click is ambiguous between overlapping people; click nearer one person's centre"
    return hits[0][1], "ok"


class OwnerTracker:
    """Track one operator-selected person while refusing identity switches."""

    def __init__(self, config: TrackerConfig, reid: PersonReID,
                 detector: Optional[PersonDetector] = None,
                 motion: Optional[MotionPredictor] = None) -> None:
        self.config = config
        self.reid = reid
        self.detector = detector
        self.motion = motion or KalmanMotionPredictor(config.motion)
        self.identity = IdentityManager(config.identity, reid)
        self.scorer = AssociationScorer(config.association, config.motion)
        self.confidence_manager = ConfidenceManager(config.confidence, config.state)
        self.tracklets = ShortTermTracklets(config.motion)
        self.frame_index = -1
        self._current_tracklet_ids = np.zeros(0, dtype=np.int64)
        self._current_tracklet_ages = np.zeros(0, dtype=np.int32)
        self.last_detections: list[Detection] = []
        self.counters: Counter[str] = Counter()
        self._owner_serial = 0
        self._clear_owner_state()

    # ------------------------------------------------------------------ setup
    def _clear_owner_state(self) -> None:
        self.owner_id: Optional[str] = None
        self.state = TrackState.UNINITIALIZED
        self.identity.reset()
        self.motion.reset()
        self.confidence_manager.reset()
        self._last_bbox: Optional[np.ndarray] = None
        self._owner_velocity: Optional[np.ndarray] = None
        self._consecutive_accepts = 0
        self._run_length = 0
        self._stable_frames = 0
        self.frames_since_seen = 0
        self.missed_frames = 0
        self.ambiguous_frames = 0
        self.search_mode = False
        self.needs_confirmation = False
        self._pending: Optional[_PendingHypothesis] = None
        self._last_gallery_or_distractor_frame = -10 ** 9
        self._excluded_tracklets: set[int] = set()
        self._owner_tracklet_id: Optional[int] = None
        self._velocity_strikes: Counter[int] = Counter()
        self._last_rival_frame = -10 ** 9

    @property
    def has_owner(self) -> bool:
        return self.owner_id is not None

    def reset(self) -> None:
        """Deliberately release the owner (operator command)."""
        if self.owner_id is not None:
            LOGGER.info("Owner %s released by operator reset", self.owner_id)
        self._clear_owner_state()
        self.counters["resets"] += 1

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.detector is None:
            raise RuntimeError("No detector configured; pass detections to update()")
        return self.detector.detect(frame)

    def select_owner(self, frame: np.ndarray, click_position: tuple[float, float],
                     detections: Optional[Sequence[Detection]] = None) -> tuple[bool, str]:
        """Lock onto the person under ``click_position``.

        Returns ``(success, message)``. Refuses while an owner is already
        locked (owner lock) - the operator must reset first.
        """
        if self.owner_id is not None:
            return False, f"owner {self.owner_id} is locked; press R to reset before re-selecting"
        dets = list(detections) if detections is not None else self.last_detections
        index, reason = find_clicked_detection(dets, click_position,
                                               self.config.identity.click_ambiguity_ratio)
        if index is None:
            return False, reason
        det = dets[index]
        embedding = self.reid.extract_embedding(frame, det.bbox)
        try:
            self.identity.initialize_owner(embedding=embedding)
        except ValueError as exc:
            return False, str(exc)
        self.motion.initialize(det.bbox)
        others = [d for j, d in enumerate(dets) if j != index
                  and iou(d.bbox, det.bbox) <= self.config.identity.distractor_isolation_iou]
        if others:
            self.identity.add_distractors(
                self.reid.extract_embeddings(frame, np.stack([d.bbox for d in others])))
        self._mark_coexisting(index, len(dets))
        self._owner_serial += 1
        self.owner_id = f"{self._owner_serial:03d}"
        self.confidence_manager.initialize(det.confidence)
        self._last_bbox = det.bbox.copy()
        self._owner_velocity = np.zeros(2)
        self._consecutive_accepts = 1
        self._run_length = 1
        self._stable_frames = 1
        self.state = TrackState.TRACKING
        self.counters["selections"] += 1
        LOGGER.info("Owner %s selected at frame %d bbox=%s", self.owner_id, self.frame_index,
                    np.round(det.bbox, 1).tolist())
        return True, f"owner {self.owner_id} selected"

    # ------------------------------------------------------------- main loop
    def update(self, frame: np.ndarray, detections: Optional[Sequence[Detection]] = None,
               timestamp: Optional[float] = None) -> FrameResult:
        """Process one frame. ``detections`` may be supplied to bypass the detector."""
        start = time.perf_counter()
        self.frame_index += 1
        timestamp = time.time() if timestamp is None else timestamp
        dets = list(detections) if detections is not None else self.detect(frame)
        self.last_detections = dets
        bboxes = (np.stack([d.bbox for d in dets]).astype(np.float32)
                  if dets else np.zeros((0, 4), np.float32))
        velocities, tracklet_ages, tracklet_ids = self.tracklets.update(bboxes)
        self._current_tracklet_ages = tracklet_ages
        self._current_tracklet_ids = tracklet_ids
        # Forget exclusions for tracklets that no longer exist.
        live = self.tracklets.live_ids
        self._excluded_tracklets &= live
        self._velocity_strikes = Counter({k: v for k, v in self._velocity_strikes.items() if k in live})

        if self.owner_id is None:
            return self._result(timestamp, start, Decision.NO_OWNER, None, None, None,
                                AssociationOutcome(Decision.NO_OWNER, None, 0.0, 0.0, None),
                                [], dets, False, False)

        cfg = self.config
        # 1. Motion prediction (stops after max_prediction_frames without evidence).
        prediction_valid = (self.motion.is_active
                            and self.frames_since_seen <= cfg.state.max_prediction_frames)
        predicted: Optional[np.ndarray] = None
        if prediction_valid:
            if self.search_mode:
                self.motion.damp_velocity(cfg.motion.velocity_damping_when_lost)
            predicted = self.motion.predict()
        search = self.search_mode
        d2 = self.motion.mahalanobis(bboxes) if predicted is not None else np.full(len(dets), np.inf)
        gate_d2 = self._gate_d2(search, predicted is not None)

        # 2. Appearance evidence.
        app_sims, distractor_sims, embeddings = self._appearance(frame, bboxes, d2, gate_d2, search)

        # 3. Multi-factor scores + hard gates.
        weights = cfg.association.reid_weights if search else cfg.association.tracking_weights
        owner_velocity = self._owner_velocity if self._velocity_reliable() else None
        candidates = self.scorer.score(dets, app_sims, distractor_sims, predicted, self._last_bbox,
                                       d2, velocities, owner_velocity, weights)
        min_app = cfg.identity.reid_threshold if search else cfg.association.min_appearance_similarity
        excluded = [int(t) in self._excluded_tracklets for t in tracklet_ids]
        reach_center, reach_radius = self._reachability(predicted is not None)
        self.scorer.apply_gates(
            candidates, gate_d2=gate_d2, min_appearance_similarity=min_app,
            enforce_velocity=owner_velocity is not None,
            min_motion_score=cfg.identity.reid_min_motion_score if (search and predicted is not None) else 0.0,
            excluded=excluded, reach_center=reach_center, reach_radius=reach_radius)

        self._record_velocity_strikes(candidates, tracklet_ids)

        # 4. Threshold + margin (anti-switch rule 1 and 2).
        threshold = cfg.identity.reid_score_threshold if search else cfg.identity.identity_threshold
        margin = cfg.identity.reid_min_score_margin if search else cfg.identity.min_score_margin
        outcome = evaluate_candidates(candidates, threshold, margin)

        # 5. Temporal confirmation (anti-switch rule 3) and state update.
        decision, accepted, continuous = self._resolve(outcome, predicted, search, tracklet_ids,
                                                        candidates)
        reidentified = False
        if decision is Decision.ACCEPT and accepted is not None:
            reidentified = self._on_accept(accepted, outcome, dets, embeddings, search, frame)
        switch_attempt = self._is_switch_attempt(decision, outcome, continuous, threshold)
        if switch_attempt:
            self.counters["identity_switch_attempts_blocked"] += 1
        owner_bbox = accepted.bbox.copy() if (decision is Decision.ACCEPT and accepted is not None
                                              and self.state in (TrackState.TRACKING,
                                                                 TrackState.LOW_CONFIDENCE)) else None
        return self._result(timestamp, start, decision, owner_bbox, predicted,
                            accepted.index if accepted is not None and decision is Decision.ACCEPT else None,
                            outcome, candidates, dets, switch_attempt, reidentified)

    # ----------------------------------------------------------- components
    def _gate_d2(self, search: bool, has_prediction: bool) -> float:
        mc = self.config.motion
        if not has_prediction:
            return math.inf
        if search:
            multiplier = mc.max_gate_multiplier
        else:
            multiplier = min(mc.max_gate_multiplier, 1.0 + mc.gate_growth_per_frame * self.frames_since_seen)
        return mc.gating_chi2 * multiplier ** 2

    def _reachability(self, has_prediction: bool) -> tuple[Optional[tuple[float, float]], float]:
        """Disc the owner can physically have reached since last confirmed.

        Only enforced while a motion prediction exists; after a long absence
        the owner may re-enter anywhere.
        """
        if not has_prediction or self._last_bbox is None or self.frames_since_seen == 0:
            return None, math.inf
        mc = self.config.motion
        h = max(float(self._last_bbox[3] - self._last_bbox[1]), 1.0)
        radius = h * (mc.reachability_base_heights
                      + mc.max_speed_heights_per_frame * (self.frames_since_seen + 1))
        return bbox_center(self._last_bbox), radius

    def _velocity_reliable(self) -> bool:
        """Owner velocity is trusted if it came from a long enough run of
        confirmed frames and the owner was seen recently."""
        mc = self.config.motion
        return (self._owner_velocity is not None
                and self._run_length >= mc.min_frames_for_reliable_velocity
                and self.frames_since_seen <= mc.velocity_memory_frames)

    def _appearance(self, frame: np.ndarray, bboxes: np.ndarray, d2: np.ndarray, gate_d2: float,
                    search: bool) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        n = len(bboxes)
        if n == 0:
            return np.zeros(0), np.zeros(0), None
        if self.config.reid.prefilter_by_motion_gate and not search and math.isfinite(gate_d2):
            idx = np.flatnonzero(d2 <= gate_d2 * 4.0)
        else:
            idx = np.arange(n)
        embeddings = np.zeros((n, max(self.reid.embedding_dim, 1)), dtype=np.float32)
        if len(idx):
            embeddings[idx] = self.reid.extract_embeddings(frame, bboxes[idx])
        valid = np.any(embeddings != 0, axis=1)
        app = self.identity.similarity(embeddings).astype(np.float64)
        app[~valid] = np.nan
        distractor = self.identity.distractor_similarity(embeddings)
        return app, distractor, embeddings

    def _resolve(self, outcome: AssociationOutcome, predicted: Optional[np.ndarray], search: bool,
                 tracklet_ids: np.ndarray, outcome_candidates: Sequence[CandidateScore] = ()
                 ) -> tuple[Decision, Optional[CandidateScore], bool]:
        """Turn the per-frame outcome into a decision using temporal memory.

        Fast path (immediate ACCEPT) requires *all* of: normal tracking mode,
        no pending need for confirmation, spatial continuity with the
        prediction, and an unbroken detection-level link to the owner's
        previous detection. Anything else - a jump, a broken link (typical
        at crossings), recovery from ambiguity or loss - must survive
        ``confirmation_frames`` consecutive frames first.
        """
        cfg = self.config
        if outcome.decision is Decision.MISS:
            self.handle_occlusion(ambiguous=False)
            return Decision.MISS, None, False
        if outcome.decision is Decision.AMBIGUOUS:
            self.handle_occlusion(ambiguous=True)
            return Decision.AMBIGUOUS, None, False

        best = outcome.best
        assert best is not None
        continuous = predicted is not None and (
            iou(predicted, best.bbox) >= cfg.identity.continuity_iou
            or best.mahalanobis_d2 <= cfg.motion.gating_chi2)
        same_link = (self._owner_tracklet_id is not None and len(tracklet_ids) > best.index
                     and int(tracklet_ids[best.index]) == self._owner_tracklet_id)
        if continuous and same_link and not search and not self.needs_confirmation:
            return Decision.ACCEPT, best, True

        # Evidence must persist across consecutive frames before it is trusted.
        required = cfg.identity.reid_confirmation_frames if search else cfg.identity.confirmation_frames
        # The hypothesis continues on the same detection-level tracklet, or on a
        # brand-new tracklet at the same place (the linker refused a link, which
        # it does whenever unsure). It never continues on a *different,
        # established* tracklet: at a crossing another person can overlap the
        # pending box, and IoU alone would let them inherit the evidence.
        pending = self._pending
        has_link = len(tracklet_ids) > best.index
        best_tracklet = int(tracklet_ids[best.index]) if has_link else -1
        fresh = has_link and len(self._current_tracklet_ages) > best.index \
            and int(self._current_tracklet_ages[best.index]) == 0
        rival_recent = (self.frame_index - self._last_rival_frame
                        <= cfg.identity.confirmation_frames)
        if (pending is not None
                and (best_tracklet == pending.tracklet_id or (fresh and not rival_recent))
                and iou(pending.bbox, best.bbox) >= cfg.identity.pending_match_iou):
            pending.tracklet_id = best_tracklet
            pending.count += 1
            pending.bbox = best.bbox.copy()
        else:
            pending = self._pending = _PendingHypothesis(best.bbox.copy(), 1, self.frame_index,
                                                         best_tracklet)
        # Remember when any other person was close to the hypothesis: for a
        # while afterwards a fresh tracklet may be that person rather than a
        # broken link, so evidence is not carried over to fresh tracklets.
        h = max(float(best.bbox[3] - best.bbox[1]), 1.0)
        bx, by = bbox_center(best.bbox)
        if any(c.index != best.index
               and math.hypot(bbox_center(c.bbox)[0] - bx, bbox_center(c.bbox)[1] - by) < h
               for c in outcome_candidates):
            self._last_rival_frame = self.frame_index
        # While the owner's velocity is trusted, a candidate must also demonstrate
        # compatible motion (known velocity) before confirmation.
        velocity_ok = math.isfinite(best.velocity_z) if self._velocity_reliable() else True
        if pending.count >= required and velocity_ok:
            return Decision.ACCEPT, best, continuous
        self._on_pending()
        return Decision.PENDING, best, continuous

    def handle_occlusion(self, ambiguous: bool) -> None:
        """Bookkeeping for a frame with no trusted owner measurement.

        The Kalman filter keeps coasting (done in :meth:`update`), the appearance
        model is kept unchanged, the search region grows with time, and no
        other person is allowed to take over the identity.
        """
        cc = self.config.confidence
        self.frames_since_seen += 1
        self._pending = None
        self._consecutive_accepts = 0
        if ambiguous:
            self.ambiguous_frames += 1
            self.needs_confirmation = True
            self.counters["ambiguous_frames"] += 1
            report = self.confidence_manager.decay(cc.ambiguous_decay)
        else:
            self.missed_frames += 1
            self.counters["missed_frames"] += 1
            if self.frames_since_seen > self.config.identity.short_gap_frames:
                self.needs_confirmation = True
            report = self.confidence_manager.decay(cc.miss_decay)
        self._check_lost(report)
        self._set_unmeasured_state(TrackState.AMBIGUOUS if ambiguous else TrackState.LOW_CONFIDENCE)

    def _on_pending(self) -> None:
        self.frames_since_seen += 1
        self._consecutive_accepts = 0
        self.needs_confirmation = True
        report = self.confidence_manager.decay(self.config.confidence.ambiguous_decay)
        self._check_lost(report)
        self.state = TrackState.REIDENTIFYING

    def _check_lost(self, report: ConfidenceReport) -> None:
        s = self.config.state
        if self.search_mode:
            return
        if (self.missed_frames >= s.lost_after_missed_frames
                or self.frames_since_seen >= s.lost_after_unresolved_frames
                or report.overall < s.ambiguous_threshold):
            self.search_mode = True
            self.needs_confirmation = True
            self.counters["lost_events"] += 1
            LOGGER.info("Owner %s LOST at frame %d (missed=%d, unresolved=%d, conf=%.2f)",
                        self.owner_id, self.frame_index, self.missed_frames,
                        self.frames_since_seen, report.overall)

    def _set_unmeasured_state(self, decision_state: TrackState) -> None:
        if self.search_mode:
            self.state = TrackState.LOST
            return
        conf_state = self.confidence_manager.state_for(self.confidence_manager.report.overall)
        if conf_state is TrackState.TRACKING:
            conf_state = TrackState.LOW_CONFIDENCE   # never TRACKING without a measurement
        self.state = max(decision_state, conf_state, key=STATE_SEVERITY.__getitem__)

    def _on_accept(self, best: CandidateScore, outcome: AssociationOutcome,
                   dets: Sequence[Detection], embeddings: Optional[np.ndarray],
                   search: bool, frame: np.ndarray) -> bool:
        cfg = self.config
        reacquired = search
        was_confirmed = self.needs_confirmation
        long_gap = (search or self.frames_since_seen > cfg.state.max_prediction_frames
                    or not math.isfinite(best.mahalanobis_d2)
                    or best.mahalanobis_d2 > cfg.motion.gating_chi2 * cfg.motion.max_gate_multiplier ** 2)
        self._consecutive_accepts += 1
        if long_gap:
            # Prediction is stale: restart the motion model and its velocity history.
            self.motion.initialize(best.bbox)
            self._owner_velocity = None
            self._consecutive_accepts = 1
        else:
            self.motion.update(best.bbox)
            self._owner_velocity = self.motion.velocity
        self._run_length = self._consecutive_accepts
        self._last_bbox = best.bbox.copy()
        self._mark_coexisting(best.index, len(dets))
        self._stable_frames = 1 if (reacquired or was_confirmed) else self._stable_frames + 1
        self.frames_since_seen = 0
        self.missed_frames = 0
        self.ambiguous_frames = 0
        self.search_mode = False
        self.needs_confirmation = False
        self._pending = None

        spatial = np.nanmean([best.iou_score, best.position_score]) \
            if math.isfinite(best.iou_score) or math.isfinite(best.position_score) else math.nan
        report = self.confidence_manager.calculate_confidence(
            appearance=best.appearance_score, motion=best.motion_score, spatial=float(spatial),
            detection=best.detection_confidence, association_score=best.total, margin=outcome.margin)
        conf_state = self.confidence_manager.state_for(report.overall)
        if conf_state in (TrackState.TRACKING, TrackState.LOW_CONFIDENCE):
            self.state = conf_state
        else:
            # Evidence accepted (motion model updated) but smoothed confidence is
            # still too low to report this box as the owner.
            self.state = TrackState.AMBIGUOUS

        if reacquired:
            self.counters["reidentifications"] += 1
            LOGGER.info("Owner %s re-identified at frame %d (score=%.2f, sim=%.2f)", self.owner_id,
                        self.frame_index, best.total, best.appearance_similarity)
        if self.state is TrackState.TRACKING and embeddings is not None:
            self._update_appearance_models(best, outcome, dets, embeddings, report.overall)
        return reacquired

    def _record_velocity_strikes(self, candidates: Sequence[CandidateScore],
                                 tracklet_ids: np.ndarray) -> None:
        """A tracklet that repeatedly moves against the owner's known trajectory
        is excluded for as long as it stays unbroken - even after the owner's
        velocity memory expires. This is what prevents a look-alike who crossed
        the owner from being adopted once the owner has left the view."""
        limit = self.config.motion.velocity_strikes_to_exclude
        if limit <= 0:
            return
        for c in candidates:
            if c.rejected_reason == "velocity_gate":
                tid = int(tracklet_ids[c.index])
                self._velocity_strikes[tid] += 1
                if self._velocity_strikes[tid] >= limit and tid not in self._excluded_tracklets:
                    self._excluded_tracklets.add(tid)
                    self.counters["tracklets_excluded_by_motion"] += 1

    def _mark_coexisting(self, owner_index: int, n: int) -> None:
        """Every other detection visible alongside the confirmed owner is a
        different person; remember its tracklet as excluded."""
        ids = self._current_tracklet_ids
        if len(ids) != n:
            return
        self._owner_tracklet_id = int(ids[owner_index])
        self._excluded_tracklets.update(int(ids[j]) for j in range(n) if j != owner_index)
        self._excluded_tracklets.discard(int(ids[owner_index]))

    def _update_appearance_models(self, best: CandidateScore, outcome: AssociationOutcome,
                                  dets: Sequence[Detection], embeddings: np.ndarray,
                                  confidence: float) -> None:
        ic = self.config.identity
        others = [j for j in range(len(dets)) if j != best.index]
        overlaps = (iou_matrix(best.bbox[None], np.stack([dets[j].bbox for j in others]))[0]
                    if others else np.zeros(0))
        isolated = bool(np.all(overlaps < ic.gallery_isolation_iou))
        if self.identity.maybe_update(
                embeddings[best.index], confidence=confidence, similarity=best.appearance_similarity,
                margin=outcome.margin, isolated=isolated, stable_frames=self._stable_frames,
                frame_index=self.frame_index):
            self.counters["gallery_updates"] += 1
        if (confidence >= ic.gallery_update_confidence and others
                and self.frame_index - self._last_gallery_or_distractor_frame >= ic.gallery_update_interval):
            clean = [j for j, ov in zip(others, overlaps) if ov <= ic.distractor_isolation_iou]
            if clean:
                self.identity.add_distractors(embeddings[clean])
                self._last_gallery_or_distractor_frame = self.frame_index

    def _is_switch_attempt(self, decision: Decision, outcome: AssociationOutcome,
                           continuous: bool, threshold: float) -> bool:
        """Would a naive tracker have jumped to a different person this frame?"""
        if decision is Decision.AMBIGUOUS:
            return True
        top = outcome.top_overall
        if top is not None and not top.passed and top.total >= threshold:
            return True
        return decision is Decision.PENDING and not continuous

    def _result(self, timestamp: float, start: float, decision: Decision,
                owner_bbox: Optional[np.ndarray], predicted: Optional[np.ndarray],
                owner_index: Optional[int], outcome: AssociationOutcome,
                candidates: list[CandidateScore], dets: list[Detection],
                switch_attempt: bool, reidentified: bool) -> FrameResult:
        return FrameResult(
            frame_index=self.frame_index, timestamp=timestamp, state=self.state, decision=decision,
            owner_id=self.owner_id, owner_bbox=owner_bbox, predicted_bbox=predicted,
            owner_detection_index=owner_index, confidence=self.confidence_manager.report,
            best_candidate=outcome.best, second_best_score=outcome.second_best_score,
            score_margin=outcome.margin, identity_switch_attempt=switch_attempt,
            reidentified=reidentified, candidates=candidates, detections=dets,
            gallery_size=len(self.identity.gallery),
            processing_ms=(time.perf_counter() - start) * 1000.0)
