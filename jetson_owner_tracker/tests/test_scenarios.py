"""Crossing / occlusion / look-alike scenario tests.

The primary requirement: the tracker must never report a person other than
the owner. Losing the owner (AMBIGUOUS/LOST) is acceptable; a switch is not.
"""

from __future__ import annotations

import pytest

from owner_tracking.config import TrackerConfig
from owner_tracking.evaluation import evaluate_scenario_results
from owner_tracking.reid import HistogramReID
from owner_tracking.simulation import build_scenarios, run_scenario
from owner_tracking.tracker import OwnerTracker

SCENARIOS = list(build_scenarios(0).keys())
NOISE = [(1.5, 0.0), (4.0, 0.10)]


@pytest.mark.parametrize("name", SCENARIOS)
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("jitter,miss", NOISE)
def test_no_wrong_owner_frames(name, seed, jitter, miss):
    sc = build_scenarios(seed, jitter, miss)[name]
    results = run_scenario(sc, OwnerTracker(TrackerConfig(), HistogramReID()))
    wrong = [sf.index for sf, r in results
             if r.reported and sf.detections[r.owner_detection_index].source_id != sc.owner_id]
    assert not wrong, f"{name}: wrong owner reported at frames {wrong[:10]}"
    m = evaluate_scenario_results(results, sc.owner_id)
    assert m.incorrect_owner_switches == 0 and m.false_owner_frames == 0


@pytest.mark.parametrize("name", [n for n, s in build_scenarios(0).items() if s.expect_recovery])
@pytest.mark.parametrize("seed", range(4))
def test_owner_recovered_after_occlusion(name, seed):
    sc = build_scenarios(seed)[name]
    results = run_scenario(sc, OwnerTracker(TrackerConfig(), HistogramReID()))
    recovered = [sf.index for sf, r in results
                 if sf.index >= sc.recovery_after_frame and r.reported
                 and sf.detections[r.owner_detection_index].source_id == sc.owner_id]
    assert recovered, f"{name}: owner never re-acquired"


def test_state_machine_visits_non_tracking_states_during_crossing():
    sc = build_scenarios(0)["crossing_similar"]
    states = {r.state.value for _, r in run_scenario(sc, OwnerTracker(TrackerConfig(), HistogramReID()))}
    assert "TRACKING" in states
    assert states & {"AMBIGUOUS", "LOST", "REIDENTIFYING", "LOW_CONFIDENCE"}


def test_deterministic():
    runs = []
    for _ in range(2):
        sc = build_scenarios(3, 3.0, 0.05)["occlusion_with_intruders"]
        runs.append([(r.state, None if r.owner_bbox is None else tuple(r.owner_bbox))
                     for _, r in run_scenario(sc, OwnerTracker(TrackerConfig(), HistogramReID()))])
    assert runs[0] == runs[1]


@pytest.mark.parametrize("name", SCENARIOS)
@pytest.mark.parametrize("step", [3, 4])
def test_no_wrong_owner_at_low_frame_rate(name, step):
    """Keep every ``step``-th frame (~10 / ~7.5 fps) with the low-fps profile."""
    from pathlib import Path
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "jetson_low_fps.yaml"
    for seed in range(3):
        sc = build_scenarios(seed, 3.0, 0.05)[name]
        tracker = OwnerTracker(TrackerConfig.from_yaml(cfg_path), HistogramReID())
        for sf in sc.scene:
            if sf.index % step:
                continue
            r = tracker.update(sf.frame, sf.detections)
            if not tracker.has_owner and any(d.source_id == sc.owner_id for d in sf.detections):
                b = sf.ground_truth[sc.owner_id]
                tracker.select_owner(sf.frame, ((b[0] + b[2]) / 2, b[1] + 0.4 * (b[3] - b[1])),
                                     sf.detections)
                continue
            assert not (r.reported and sf.detections[r.owner_detection_index].source_id != sc.owner_id), \
                f"{name} seed {seed}: wrong owner at frame {sf.index}"
