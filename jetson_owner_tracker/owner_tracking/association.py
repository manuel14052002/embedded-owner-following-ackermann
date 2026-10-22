"""Multi-factor identity association and the pure decision rule.

Scoring (``AssociationScorer.score``) turns raw evidence into normalised
components; gating (``apply_gates``) applies hard physical/appearance
constraints; ``evaluate_candidates`` applies the identity threshold and the
best-vs-second margin rule. Temporal confirmation lives in the tracker because
it needs memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from owner_tracking.config import AssociationConfig, AssociationWeights, MotionConfig
from owner_tracking.datatypes import CandidateScore, Decision, Detection
from owner_tracking.geometry import bbox_center, iou

COMPONENTS = ("appearance", "iou", "position", "scale", "motion", "detection")


def chi2_sf_4dof(d2: float) -> float:
    """Survival function of a chi-square(4) variable: P(X >= d2).

    For 4 degrees of freedom it has the closed form ``exp(-d/2) (1 + d/2)``,
    which turns a squared Mahalanobis distance into a calibrated [0, 1]
    "how plausible is this position" score.
    """
    if not math.isfinite(d2):
        return 0.0
    d2 = max(0.0, d2)
    return float(math.exp(-0.5 * d2) * (1.0 + 0.5 * d2))


def weighted_score(components: dict[str, float], weights: AssociationWeights) -> float:
    """Weighted mean over *available* (finite) components; result in [0, 1]."""
    w = weights.as_dict()
    num = den = 0.0
    for name in COMPONENTS:
        value = components.get(name, math.nan)
        weight = w[name]
        if weight <= 0.0 or not math.isfinite(value):
            continue
        num += weight * min(1.0, max(0.0, value))
        den += weight
    return num / den if den > 0 else 0.0


class AssociationScorer:
    """Compute normalised evidence components for each candidate."""

    def __init__(self, config: AssociationConfig, motion_config: MotionConfig) -> None:
        self.config = config
        self.motion_config = motion_config

    def normalize_appearance(self, similarity: float) -> float:
        """Map raw cosine similarity to [0, 1] with a calibrated floor/ceiling."""
        c = self.config
        span = c.appearance_similarity_ceiling - c.appearance_similarity_floor
        return float(np.clip((similarity - c.appearance_similarity_floor) / span, 0.0, 1.0))

    def velocity_z(self, candidate_velocity: np.ndarray, owner_velocity: Optional[np.ndarray],
                   height: float) -> float:
        """Normalised disagreement between candidate and owner velocity (nan if unknown)."""
        if owner_velocity is None or not np.all(np.isfinite(candidate_velocity)):
            return math.nan
        mc = self.motion_config
        sigma = mc.velocity_sigma * height + mc.velocity_sigma_min_px
        return float(np.linalg.norm(candidate_velocity - owner_velocity) / sigma)

    def score(
        self,
        detections: Sequence[Detection],
        appearance_similarities: np.ndarray,
        distractor_similarities: np.ndarray,
        predicted_bbox: Optional[np.ndarray],
        reference_bbox: Optional[np.ndarray],
        mahalanobis_d2: np.ndarray,
        candidate_velocities: np.ndarray,
        owner_velocity: Optional[np.ndarray],
        weights: AssociationWeights,
    ) -> list[CandidateScore]:
        """Score every detection against the owner hypothesis."""
        cfg = self.config
        scale_ref = predicted_bbox if predicted_bbox is not None else reference_bbox
        results: list[CandidateScore] = []
        for i, det in enumerate(detections):
            box = det.bbox
            sim = float(appearance_similarities[i])
            app = self.normalize_appearance(sim) if math.isfinite(sim) else math.nan
            iou_s = pos_s = motion_s = math.nan
            vz = math.nan
            d2 = float(mahalanobis_d2[i]) if len(mahalanobis_d2) else math.inf
            if predicted_bbox is not None:
                iou_s = iou(predicted_bbox, box)
                pcx, pcy = bbox_center(predicted_bbox)
                cx, cy = bbox_center(box)
                ph = max(float(predicted_bbox[3] - predicted_bbox[1]), 1.0)
                dist = math.hypot(cx - pcx, cy - pcy) / (cfg.position_sigma * ph)
                pos_s = math.exp(-0.5 * dist * dist)
                vz = self.velocity_z(candidate_velocities[i], owner_velocity, ph)
                parts = [chi2_sf_4dof(d2)]
                if math.isfinite(vz):
                    parts.append(math.exp(-0.5 * vz * vz))
                motion_s = float(np.prod(parts) ** (1.0 / len(parts)))  # geometric mean
            scale_s = math.nan
            if scale_ref is not None:
                rw = max(float(scale_ref[2] - scale_ref[0]), 1.0)
                rh = max(float(scale_ref[3] - scale_ref[1]), 1.0)
                cw = max(float(box[2] - box[0]), 1.0)
                ch = max(float(box[3] - box[1]), 1.0)
                log_err = math.log(ch / rh) ** 2 + math.log(cw / rw) ** 2
                scale_s = math.exp(-log_err / (2 * cfg.scale_sigma ** 2))
            components = {
                "appearance": app, "iou": iou_s, "position": pos_s,
                "scale": scale_s, "motion": motion_s, "detection": det.confidence,
            }
            results.append(CandidateScore(
                index=i, bbox=box, detection_confidence=det.confidence,
                appearance_similarity=sim,
                distractor_similarity=float(distractor_similarities[i]),
                appearance_score=app, iou_score=iou_s, position_score=pos_s,
                scale_score=scale_s, motion_score=motion_s, velocity_z=vz,
                mahalanobis_d2=d2, total=weighted_score(components, weights),
            ))
        return results

    def apply_gates(
        self,
        candidates: Sequence[CandidateScore],
        *,
        gate_d2: float,
        min_appearance_similarity: float,
        enforce_velocity: bool,
        min_motion_score: float = 0.0,
        excluded: Optional[Sequence[bool]] = None,
        reach_center: Optional[tuple[float, float]] = None,
        reach_radius: float = math.inf,
    ) -> None:
        """Mark physically or visually implausible candidates as rejected.

        Gates are *hard*: a rejected candidate can never become the owner in
        this frame regardless of its detector confidence or total score.
        Order: co-existence exclusion, physical reachability, embedding
        validity, appearance floor, distractor resemblance, Kalman gate,
        velocity gate, minimum motion plausibility.
        """
        mc = self.motion_config
        for c in candidates:
            if excluded is not None and excluded[c.index]:
                c.rejected_reason = "coexisted_with_owner"
            elif reach_center is not None and math.isfinite(reach_radius) and math.hypot(
                    bbox_center(c.bbox)[0] - reach_center[0],
                    bbox_center(c.bbox)[1] - reach_center[1]) > reach_radius:
                c.rejected_reason = "unreachable"
            elif not math.isfinite(c.appearance_similarity):
                c.rejected_reason = "invalid_embedding"
            elif c.appearance_similarity < min_appearance_similarity:
                c.rejected_reason = "appearance_floor"
            elif (c.distractor_similarity >= min_appearance_similarity
                  and c.distractor_similarity >= c.appearance_similarity + self.config.distractor_margin):
                c.rejected_reason = "resembles_distractor"
            elif math.isfinite(gate_d2) and c.mahalanobis_d2 > gate_d2:
                c.rejected_reason = "motion_gate"
            elif enforce_velocity and math.isfinite(c.velocity_z) and c.velocity_z > mc.velocity_gate_z:
                c.rejected_reason = "velocity_gate"
            elif min_motion_score > 0 and math.isfinite(c.motion_score) and c.motion_score < min_motion_score:
                c.rejected_reason = "implausible_motion"


@dataclass
class AssociationOutcome:
    """Result of the threshold + margin rule for one frame."""

    decision: Decision                  # ACCEPT here means "identified candidate"
    best: Optional[CandidateScore]
    second_best_score: float
    margin: float
    top_overall: Optional[CandidateScore]   # best total including rejected ones


def evaluate_candidates(candidates: Sequence[CandidateScore], identity_threshold: float,
                        min_margin: float) -> AssociationOutcome:
    """Apply the identity threshold and the ambiguity margin.

    * No gated-in candidate, or best score below the threshold -> ``MISS``.
    * Best minus second-best below ``min_margin`` -> ``AMBIGUOUS`` (choose
      neither; the competitor counts even if it is itself below threshold).
    * Otherwise -> ``ACCEPT`` (subject to temporal confirmation upstream).
    """
    top_overall = max(candidates, key=lambda c: c.total) if candidates else None
    passed = sorted((c for c in candidates if c.passed), key=lambda c: c.total, reverse=True)
    if not passed:
        return AssociationOutcome(Decision.MISS, None, 0.0, 0.0, top_overall)
    best = passed[0]
    second = passed[1].total if len(passed) > 1 else 0.0
    margin = best.total - second
    if best.total < identity_threshold:
        return AssociationOutcome(Decision.MISS, best, second, margin, top_overall)
    if margin < min_margin:
        return AssociationOutcome(Decision.AMBIGUOUS, best, second, margin, top_overall)
    return AssociationOutcome(Decision.ACCEPT, best, second, margin, top_overall)
