"""Tests for the owner-centric evaluation metrics and CSV logging."""

from __future__ import annotations

import numpy as np
import pytest

from owner_tracking.evaluation import evaluate_owner_track, load_ground_truth, load_tracker_log

A = np.array([0, 0, 10, 20], np.float32)
B = np.array([50, 0, 60, 20], np.float32)


def test_perfect_track():
    gt = [{1: A, 2: B}] * 10
    m = evaluate_owner_track(gt, [A] * 10, owner_id=1)
    assert m.tp == 10 and m.fp == m.fn == 0
    assert m.precision == m.recall == m.idf1 == m.mota == 1.0
    assert m.hota == pytest.approx(1.0)
    assert m.incorrect_owner_switches == 0 and m.loss_episodes == 0


def test_wrong_person_counts_as_switch_fp_and_fn():
    gt = [{1: A, 2: B}] * 10
    preds = [A] * 5 + [B] * 3 + [A] * 2
    m = evaluate_owner_track(gt, preds, owner_id=1)
    assert m.incorrect_owner_switches == 1
    assert m.false_owner_frames == 3 and m.fp == 3 and m.fn == 3
    assert m.id_switches == 2
    assert m.mota == pytest.approx(1 - (3 + 3 + 2) / 10)
    assert m.reid_success_rate != 1.0     # episode contained a wrong report


def test_loss_and_reid_success():
    gt = [{1: A}] * 5 + [{}] * 10 + [{1: A}] * 10
    preds = [A] * 5 + [None] * 10 + [None] * 3 + [A] * 7
    m = evaluate_owner_track(gt, preds, owner_id=1, min_reid_gap=5)
    assert m.fp == 0 and m.fn == 3 and m.recall == pytest.approx(12 / 15)
    assert m.loss_episodes == 1 and m.max_loss_frames == 13
    assert m.reid_success_rate == 1.0 and m.mean_reacquisition_frames == 3


def test_unrecovered_loss_is_failure():
    gt = [{1: A}] * 20
    m = evaluate_owner_track(gt, [A] * 5 + [None] * 15, owner_id=1)
    assert m.reid_opportunities == 1 and m.reid_success_rate == 0.0


def test_loaders_roundtrip(tmp_path):
    from owner_tracking.datatypes import ConfidenceReport, Decision, FrameResult, TrackState
    from owner_tracking.logging_utils import CSV_COLUMNS, CsvFrameLogger
    gt_path = tmp_path / "gt.csv"
    gt_path.write_text("frame,person_id,x1,y1,x2,y2\n0,1,0,0,10,20\n1,1,0,0,10,20\n1,2,50,0,60,20\n")
    log_path = tmp_path / "log.csv"
    with CsvFrameLogger(log_path) as lg:
        for f, box in enumerate([A, None]):
            lg.log(FrameResult(f, 0.0, TrackState.TRACKING, Decision.ACCEPT, "001", box, None, 0,
                               ConfidenceReport(), None, 0.0, 0.0, False, False, processing_ms=2.0))
    header = log_path.read_text().splitlines()[0].split(",")
    for col in ["timestamp", "frame_number", "owner_bbox", "detection_confidence",
                "appearance_similarity", "IoU", "motion_score", "association_score",
                "tracking_state", "identity_switch_attempt"]:
        assert col in header
    assert header == CSV_COLUMNS
    preds, lat = load_tracker_log(log_path)
    gt = load_ground_truth(gt_path, num_frames=len(preds))
    m = evaluate_owner_track(gt, preds, owner_id=1, latencies_ms=lat)
    assert m.tp == 1 and m.fn == 1 and m.fps == pytest.approx(500.0)
