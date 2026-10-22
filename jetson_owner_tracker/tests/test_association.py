"""Unit tests for association scoring, anti-switch rules and core components."""

from __future__ import annotations

import math

import numpy as np
import pytest

from owner_tracking.association import (
    AssociationScorer, chi2_sf_4dof, evaluate_candidates, weighted_score)
from owner_tracking.config import AssociationWeights, IdentityConfig, TrackerConfig
from owner_tracking.confidence import ConfidenceManager
from owner_tracking.datatypes import CandidateScore, Decision, Detection, TrackState
from owner_tracking.identity import IdentityManager
from owner_tracking.motion import KalmanMotionPredictor
from owner_tracking.reid import HistogramReID
from owner_tracking.simulation import build_scenarios, run_scenario
from owner_tracking.tracker import OwnerTracker, find_clicked_detection


def cand(index: int, total: float, **kw) -> CandidateScore:
    base = dict(bbox=np.zeros(4, np.float32), detection_confidence=0.9, appearance_similarity=0.8,
                distractor_similarity=0.0, appearance_score=0.8, iou_score=0.5, position_score=0.5,
                scale_score=1.0, motion_score=0.5, velocity_z=math.nan, mahalanobis_d2=1.0)
    base.update(kw)
    return CandidateScore(index=index, total=total, **base)


@pytest.fixture
def scorer() -> AssociationScorer:
    cfg = TrackerConfig()
    return AssociationScorer(cfg.association, cfg.motion)


# ---------------------------------------------------------------- scoring
def test_default_weights_match_spec_and_sum_to_one():
    w = AssociationWeights()
    assert w.as_dict() == pytest.approx(dict(appearance=.40, iou=.20, position=.15, scale=.10,
                                             motion=.10, detection=.05))
    assert sum(w.as_dict().values()) == pytest.approx(1.0)


def test_weighted_score_bounds_and_missing_components_renormalise():
    w = AssociationWeights()
    assert weighted_score({k: 1.0 for k in w.as_dict()}, w) == pytest.approx(1.0)
    assert weighted_score({k: 0.0 for k in w.as_dict()}, w) == pytest.approx(0.0)
    # Out-of-range inputs are clipped.
    assert 0.0 <= weighted_score({k: 7.0 for k in w.as_dict()}, w) <= 1.0
    # Unavailable (nan) components are excluded, not treated as zero.
    s = weighted_score({"appearance": 1.0, "iou": math.nan, "position": math.nan,
                        "scale": math.nan, "motion": math.nan, "detection": 1.0}, w)
    assert s == pytest.approx(1.0)


def test_chi2_survival_function():
    assert chi2_sf_4dof(0.0) == pytest.approx(1.0)
    assert chi2_sf_4dof(9.4877) == pytest.approx(0.05, abs=1e-3)   # 95% gate
    assert chi2_sf_4dof(math.inf) == 0.0


def test_all_components_normalised(scorer):
    dets = [Detection([100, 100, 160, 260], 0.9), Detection([400, 100, 470, 280], 0.4)]
    pred = np.array([102, 101, 162, 261], np.float32)
    cands = scorer.score(dets, np.array([0.9, 0.3]), np.zeros(2), pred, pred,
                         np.array([0.5, 80.0]), np.zeros((2, 2)), np.zeros(2),
                         TrackerConfig().association.tracking_weights)
    for c in cands:
        for v in (c.appearance_score, c.iou_score, c.position_score, c.scale_score,
                  c.motion_score, c.total):
            assert 0.0 <= v <= 1.0
    assert cands[0].total > cands[1].total


def test_high_detection_confidence_stranger_cannot_win(scorer):
    """Detector confidence is a 5% cue: a 0.99 stranger loses to a 0.4 owner."""
    pred = np.array([100, 100, 160, 260], np.float32)
    dets = [Detection(pred + 2, 0.40), Detection(pred + [30, 0, 30, 0], 0.99)]
    cands = scorer.score(dets, np.array([0.85, 0.30]), np.zeros(2), pred, pred,
                         np.array([0.2, 3.0]), np.zeros((2, 2)), np.zeros(2),
                         TrackerConfig().association.tracking_weights)
    scorer.apply_gates(cands, gate_d2=9.4877, min_appearance_similarity=0.35, enforce_velocity=False)
    assert cands[1].rejected_reason == "appearance_floor"
    out = evaluate_candidates(cands, 0.55, 0.12)
    assert out.decision is Decision.ACCEPT and out.best.index == 0


# ------------------------------------------------------------ anti-switch
def test_small_margin_is_ambiguous_and_selects_nobody():
    out = evaluate_candidates([cand(0, 0.80), cand(1, 0.75)], 0.55, 0.12)
    assert out.decision is Decision.AMBIGUOUS
    assert out.margin == pytest.approx(0.05)


def test_below_threshold_is_miss():
    out = evaluate_candidates([cand(0, 0.50)], 0.55, 0.12)
    assert out.decision is Decision.MISS


def test_clear_winner_accepted():
    out = evaluate_candidates([cand(0, 0.90), cand(1, 0.40)], 0.55, 0.12)
    assert out.decision is Decision.ACCEPT and out.best.index == 0


def test_rejected_candidates_never_win_but_are_reported():
    c = [cand(0, 0.99, rejected_reason="motion_gate"), cand(1, 0.30)]
    out = evaluate_candidates(c, 0.55, 0.12)
    assert out.decision is Decision.MISS
    assert out.top_overall.index == 0


def test_velocity_gate_rejects_opposite_motion(scorer):
    c = cand(0, 0.9, velocity_z=10.0)
    scorer.apply_gates([c], gate_d2=9.4877, min_appearance_similarity=0.35, enforce_velocity=True)
    assert c.rejected_reason == "velocity_gate"
    c2 = cand(0, 0.9, velocity_z=10.0)
    scorer.apply_gates([c2], gate_d2=9.4877, min_appearance_similarity=0.35, enforce_velocity=False)
    assert c2.passed


def test_motion_gate_and_distractor_gate(scorer):
    far = cand(0, 0.9, mahalanobis_d2=50.0)
    look = cand(1, 0.9, appearance_similarity=0.6, distractor_similarity=0.9)
    scorer.apply_gates([far, look], gate_d2=9.4877, min_appearance_similarity=0.35,
                       enforce_velocity=False)
    assert far.rejected_reason == "motion_gate"
    assert look.rejected_reason == "resembles_distractor"


def test_coexistence_exclusion(scorer):
    c = [cand(0, 0.95), cand(1, 0.95)]
    scorer.apply_gates(c, gate_d2=math.inf, min_appearance_similarity=0.35, enforce_velocity=False,
                       excluded=[True, False])
    assert c[0].rejected_reason == "coexisted_with_owner" and c[1].passed


# ----------------------------------------------------------- selection
def test_click_outside_rejected_and_inside_selected():
    dets = [Detection([100, 100, 160, 260], 0.9), Detection([300, 100, 360, 260], 0.9)]
    assert find_clicked_detection(dets, (10, 10), 0.15)[0] is None
    assert find_clicked_detection(dets, (330, 150), 0.15)[0] == 1


def test_click_between_overlapping_people_is_ambiguous():
    dets = [Detection([100, 100, 160, 260], 0.9), Detection([102, 100, 162, 260], 0.9)]
    idx, reason = find_clicked_detection(dets, (131, 180), 0.15)
    assert idx is None and "ambiguous" in reason


def test_owner_lock_refuses_reselection_until_reset():
    sc = build_scenarios(0)["crossing_distinct"]
    tracker = OwnerTracker(TrackerConfig(), HistogramReID())
    sf = next(iter(sc.scene))
    tracker.update(sf.frame, sf.detections)
    owner_box, other_box = sf.ground_truth[1], sf.ground_truth[2]
    ok, _ = tracker.select_owner(sf.frame, tuple((owner_box[:2] + owner_box[2:]) / 2), sf.detections)
    assert ok and tracker.owner_id == "001"
    ok2, msg = tracker.select_owner(sf.frame, tuple((other_box[:2] + other_box[2:]) / 2),
                                    sf.detections)
    assert not ok2 and "locked" in msg and tracker.owner_id == "001"
    tracker.reset()
    assert tracker.owner_id is None and tracker.state is TrackState.UNINITIALIZED
    ok3, _ = tracker.select_owner(sf.frame, tuple((other_box[:2] + other_box[2:]) / 2), sf.detections)
    assert ok3 and tracker.owner_id == "002"


def test_reidentification_requires_temporal_confirmation():
    """After a long absence the returning owner is PENDING for several frames first."""
    cfg = TrackerConfig()
    sc = build_scenarios(0)["long_term_reid"]
    results = run_scenario(sc, OwnerTracker(cfg, HistogramReID()))
    returned = [i for i, (sf, _) in enumerate(results) if i >= 100 and 1 in sf.ground_truth]
    first_accept = next(i for i in returned if results[i][1].reported)
    pending = [results[i][1].decision for i in returned if i < first_accept]
    assert first_accept - returned[0] >= cfg.identity.reid_confirmation_frames - 1
    assert Decision.PENDING in pending


# ------------------------------------------------------- identity model
def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_gallery_update_safeguards():
    im = IdentityManager(IdentityConfig())
    im.initialize_owner(embedding=_unit([1, 0, 0, 0]))
    new = _unit([0.9, 0.3, 0, 0])
    ok_kwargs = dict(confidence=0.9, similarity=0.9, margin=0.5, isolated=True,
                     stable_frames=50, frame_index=100)
    for bad in (dict(confidence=0.5), dict(margin=0.05), dict(isolated=False),
                dict(stable_frames=2), dict(similarity=0.3)):
        assert not im.maybe_update(new, **{**ok_kwargs, **bad})
    assert im.maybe_update(new, **ok_kwargs)
    # Rate limited and de-duplicated.
    assert not im.maybe_update(_unit([0.8, 0, 0.6, 0]), **{**ok_kwargs, "frame_index": 101})
    assert not im.maybe_update(new, **{**ok_kwargs, "frame_index": 200})
    # The anchor always contributes: a gallery can't fully redefine the owner.
    assert len(im.gallery) == 1


def test_lookalikes_not_stored_as_distractors():
    im = IdentityManager(IdentityConfig())
    im.initialize_owner(embedding=_unit([1, 0, 0, 0]))
    added = im.add_distractors(np.stack([_unit([1, 0.05, 0, 0]), _unit([0, 1, 0, 0])]))
    assert added == 1


# ------------------------------------------------------- motion / confidence
def test_kalman_constant_velocity_prediction():
    kf = KalmanMotionPredictor(TrackerConfig().motion)
    for t in range(15):
        kf.predict() if t else None
        box = np.array([100 + 5 * t, 100, 160 + 5 * t, 260], np.float32)
        if t == 0:
            kf.initialize(box)
        else:
            kf.update(box)
    pred = kf.predict()
    assert pred[0] == pytest.approx(175, abs=3.0)
    d2 = kf.mahalanobis(np.stack([pred, pred + [200, 0, 200, 0]]))
    assert d2[0] < 1.0 < 9.4877 < d2[1]


def test_confidence_bands():
    cfg = TrackerConfig()
    cm = ConfidenceManager(cfg.confidence, cfg.state)
    assert cm.state_for(0.8) is TrackState.TRACKING
    assert cm.state_for(0.6) is TrackState.LOW_CONFIDENCE
    assert cm.state_for(0.4) is TrackState.AMBIGUOUS
    assert cm.state_for(0.1) is TrackState.LOST
    cm.initialize(0.9)
    r = cm.calculate_confidence(appearance=0.2, motion=0.2, spatial=0.2, detection=0.2,
                                association_score=0.2, margin=0.0)
    assert r.overall < 1.0 and set(r.as_dict()) >= {"appearance", "motion", "spatial",
                                                   "detection", "identity", "overall"}


# ---------------------------------------------------------------- config
def test_config_yaml_roundtrip_and_unknown_key(tmp_path):
    from pathlib import Path
    default = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    assert TrackerConfig.from_yaml(default).to_dict() == TrackerConfig().to_dict()
    p = tmp_path / "c.yaml"
    p.write_text("identity:\n  min_score_margin: 0.2\n")
    assert TrackerConfig.from_yaml(p).identity.min_score_margin == 0.2
    p.write_text("identity:\n  min_score_marginn: 0.2\n")
    with pytest.raises(KeyError):
        TrackerConfig.from_yaml(p)
