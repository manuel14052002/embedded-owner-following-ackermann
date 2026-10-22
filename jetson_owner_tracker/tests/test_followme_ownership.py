"""Ownership by raised open palm: claim, auto-calibration, protection, release."""

import numpy as np

from followme.config import FollowMeConfig
from followme.gestures import FOLLOW, OTHER, WAIT
from followme.navigation import RecordingNavigation
from followme.ownership import OwnerState, OwnershipManager, hand_is_raised
from followme.pipeline import FollowMeController, FollowMePerception
from followme.reid import build_feature_extractor
from followme.segmentation import StaticSegmenter
from followme.state_machine import State
from test_followme_pipeline import INTR, OTHER as GREEN, TARGET as RED, render


def test_hand_is_raised():
    box = np.array([100, 100, 200, 500.])
    pts = np.zeros((21, 2))
    pts[0] = [150, 150]                    # wrist near the head
    assert hand_is_raised(pts, box, 0.25)
    pts[0] = [150, 300]                    # chest height -> WAIT, not raised
    assert not hand_is_raised(pts, box, 0.25)
    pts[0] = [260, 150]                    # outside the box (someone else's hand)
    assert not hand_is_raised(pts, box, 0.25)


class ScriptedOwnership(OwnershipManager):
    """Hands are scripted per clothing colour instead of detected by MediaPipe."""

    hands: dict = {}
    labels: list = []
    boxes: list = []

    def _read_hand(self, frame, bbox):
        for lab, b in zip(self.labels, self.boxes):
            if np.allclose(b, bbox):
                return self.hands.get(lab)
        return None

    def _owner_hand(self, frame, result):
        if result.target_index is None:
            return None
        return self.hands.get(self.labels[result.target_index])


def test_claim_protect_release_cycle():
    cfg = FollowMeConfig()
    cfg.reid.backend = "histogram"
    cfg.gestures.enabled = False
    cfg.calibration.metric = "gallery"
    cfg.calibration.num_samples = 40
    cfg.localization.confirm_frames = 2
    cfg.ownership.enabled = True
    cfg.ownership.calibration_s = 3.0
    cfg.ownership.owner_lost_release_s = 5.0
    seg = StaticSegmenter()
    per = FollowMePerception(cfg, seg, build_feature_extractor(cfg.reid, "cpu"), INTR)
    ctl = FollowMeController(cfg, RecordingNavigation())
    own = ScriptedOwnership(cfg, per)
    rng = np.random.default_rng(0)
    t = [0.0]
    events_seen = []

    def step(people, hands):
        t[0] += 0.1
        frame, depth, insts, labels = render(people, rng)
        seg.next = insts
        own.labels, own.boxes, own.hands = labels, [p.bbox for p in insts], hands
        r = per.process(frame, depth, t[0])
        ctl.step(r)
        for ev in own.update(frame, depth, r, t[0]):
            events_seen.append(ev)
            if ev == "follow":
                ctl.command(FOLLOW, t[0])
            elif ev in ("released", "claim_started", "claim_aborted"):
                ctl.machine.reset()
                ctl.stop(t[0])
        return r, labels

    both = [(RED, -0.8, 3.0), (GREEN, 0.8, 3.0)]
    assert own.state == OwnerState.NO_OWNER
    for _ in range(30):                                   # both raise: nobody gets it
        step(both, {RED: (WAIT, True), GREEN: (WAIT, True)})
    assert own.state == OwnerState.NO_OWNER and "ONE" in own.message
    for _ in range(10):                                   # waving briefly (< 2 s) is not a claim
        step(both, {RED: (WAIT, True)})
    for _ in range(10):
        step(both, {})
    assert own.state == OwnerState.NO_OWNER
    for _ in range(25):                                   # RED holds a raised palm > 2 s
        step(both, {RED: (WAIT, True)})
    assert own.state == OwnerState.CLAIMING
    k = 0
    while own.state == OwnerState.CLAIMING and k < 200:   # RED walks while GREEN stands aside
        z = 2.4 + 1.0 * np.sin(0.2 * k)
        step([(RED, -0.8, z), (GREEN, 1.2, 3.5)], {})
        k += 1
    assert own.state == OwnerState.OWNED
    assert len(per.memory.negatives) > 0                  # GREEN is a known stranger from the start
    for _ in range(10):
        r, labels = step([(RED, -0.8, 3.0), (GREEN, 1.2, 3.5)], {})
    assert "follow" in events_seen and ctl.state == State.FOLLOW
    assert labels[r.target_index] == RED
    for _ in range(40):                                   # GREEN tries to claim: ignored
        r, labels = step([(RED, -0.8, 3.0), (GREEN, 1.2, 3.5)], {GREEN: (WAIT, True)})
        assert r.target_index is None or labels[r.target_index] == RED
    assert own.state == OwnerState.OWNED
    for _ in range(20):                                   # low open palm = WAIT only, no release
        step([(RED, -0.8, 3.0)], {RED: (WAIT, False)})
    assert own.state == OwnerState.OWNED
    for _ in range(35):                                   # RED raises the palm 3 s -> release
        step([(RED, -0.8, 3.0), (GREEN, 1.2, 3.5)], {RED: (WAIT, True)})
    assert own.state == OwnerState.NO_OWNER and "released" in events_seen
    assert per.model is None and ctl.state == State.WAIT
    for _ in range(15):                                   # cooldown: RED's hand still up, no re-claim
        step([(RED, -0.8, 3.0)], {RED: (WAIT, True)})
    assert own.state in (OwnerState.NO_OWNER, OwnerState.CLAIMING)
    # GREEN claims now, then disappears -> auto release after owner_lost_release_s
    events_seen.clear()
    for _ in range(25):
        step([(GREEN, 0.0, 3.0)], {GREEN: (OTHER, False)})
    own.release(t[0], "test")
    for _ in range(55):                                   # 3 s cooldown + 2 s hold
        step([(GREEN, 0.0, 3.0)], {GREEN: (WAIT, True)})
    assert own.state == OwnerState.CLAIMING
    k = 0
    while own.state == OwnerState.CLAIMING and k < 200:
        step([(GREEN, 0.3 * np.sin(0.3 * k), 2.4 + 1.0 * np.sin(0.2 * k))], {})
        k += 1
    assert own.state == OwnerState.OWNED
    for _ in range(60):                                   # 6 s nobody in view
        step([], {})
    assert own.state == OwnerState.NO_OWNER and "released" in events_seen


def test_voice_armed_claim_is_quick():
    cfg = FollowMeConfig()
    cfg.reid.backend = "histogram"
    cfg.gestures.enabled = False
    cfg.ownership.enabled = True
    seg = StaticSegmenter()
    per = FollowMePerception(cfg, seg, build_feature_extractor(cfg.reid, "cpu"), INTR)
    own = ScriptedOwnership(cfg, per)
    rng = np.random.default_rng(1)
    t = [0.0]

    def step(hands):
        t[0] += 0.1
        frame, depth, insts, labels = render([(RED, 0.0, 3.0)], rng)
        seg.next = insts
        own.labels, own.boxes, own.hands = labels, [p.bbox for p in insts], hands
        return own.update(frame, depth, per.process(frame, depth, t[0]), t[0])

    for _ in range(8):                                    # 0.8 s raised: not enough normally
        step({RED: (WAIT, True)})
    assert own.state == OwnerState.NO_OWNER
    for _ in range(10):
        step({})
    assert own.arm(t[0]) and own.armed(t[0])              # voice: "follow me"
    for _ in range(7):                                    # 0.7 s raised is enough when armed
        step({RED: (WAIT, True)})
    assert own.state == OwnerState.CLAIMING and not own.armed(t[0])
    assert not own.arm(t[0])                              # cannot arm while someone claims/owns


def test_voice_arm_expires():
    cfg = FollowMeConfig()
    cfg.reid.backend = "histogram"
    cfg.gestures.enabled = False
    cfg.ownership.enabled = True
    cfg.ownership.voice_arm_s = 1.0
    per = FollowMePerception(cfg, StaticSegmenter(), build_feature_extractor(cfg.reid, "cpu"), INTR)
    own = ScriptedOwnership(cfg, per)
    own.arm(0.0)
    assert own.armed(0.5) and not own.armed(1.1)


def test_click_to_select_claims_the_clicked_person():
    cfg = FollowMeConfig()
    cfg.reid.backend = "histogram"
    cfg.gestures.enabled = False
    cfg.ownership.enabled = True
    seg = StaticSegmenter()
    per = FollowMePerception(cfg, seg, build_feature_extractor(cfg.reid, "cpu"), INTR)
    own = ScriptedOwnership(cfg, per)
    rng = np.random.default_rng(2)
    frame, depth, insts, labels = render([(RED, -0.8, 3.0), (GREEN, 0.8, 3.0)], rng)
    seg.next = insts
    own.labels, own.boxes, own.hands = labels, [p.bbox for p in insts], {}
    green = insts[labels.index(GREEN)].bbox
    cx, cy = 0.5 * (green[0] + green[2]), 0.5 * (green[1] + green[3])
    assert own.select(cx / 640, cy / 480, 0.0)                 # normalised click on GREEN
    ev = own.update(frame, depth, per.process(frame, depth, 0.1), 0.1)
    assert ev == ["claim_started"] and own.state == OwnerState.CLAIMING
    np.testing.assert_allclose(own.claimant_box, green)
    assert not own.select(10, 10, 0.2)                          # not while claiming/owned


def test_claimant_kept_when_box_changes_shape():
    cfg = FollowMeConfig()
    cfg.reid.backend = "histogram"
    cfg.gestures.enabled = False
    cfg.ownership.enabled = True
    cfg.ownership.calibration_s = 100.0                     # stay in CLAIMING for this test
    seg = StaticSegmenter()
    per = FollowMePerception(cfg, seg, build_feature_extractor(cfg.reid, "cpu"), INTR)
    own = ScriptedOwnership(cfg, per)
    rng = np.random.default_rng(3)
    frame, depth, insts, labels = render([(RED, 0.0, 3.0)], rng)
    seg.next = insts
    own.labels, own.boxes, own.hands = labels, [p.bbox for p in insts], {}
    own.select(0.5, 0.5, 0.0)
    own.update(frame, depth, per.process(frame, depth, 0.1), 0.1)
    assert own.state == OwnerState.CLAIMING
    t = 0.1
    for z in (2.2, 1.6, 1.3):                               # walks closer fast: IoU with old box < 0.2
        t += 0.2
        frame, depth, insts, labels = render([(RED, 0.1, z)], rng)
        seg.next = insts
        own.update(frame, depth, per.process(frame, depth, t), t)
        assert own.state == OwnerState.CLAIMING
    assert own._claim_seen == t                             # still matched every frame


def test_round_robin_hand_checks_still_claim_and_still_block_two_hands():
    """hands_per_frame=1: MediaPipe runs on ONE person per frame, yet a held palm still
    claims, and two simultaneous palms are still refused."""
    cfg = FollowMeConfig()
    cfg.reid.backend = "histogram"
    cfg.gestures.enabled = False
    cfg.ownership.enabled = True
    cfg.ownership.hands_per_frame = 1
    seg = StaticSegmenter()
    per = FollowMePerception(cfg, seg, build_feature_extractor(cfg.reid, "cpu"), INTR)
    own = ScriptedOwnership(cfg, per)
    rng = np.random.default_rng(5)
    checks = []
    orig = own._read_hand

    def counting_read(frame, bbox):
        checks.append(1)
        return orig(frame, bbox)
    own._read_hand = counting_read
    t = [0.0]

    def step(hands):
        t[0] += 0.1
        frame, depth, insts, labels = render([(RED, -0.8, 3.0), (GREEN, 0.8, 3.0)], rng)
        seg.next = insts
        own.labels, own.boxes, own.hands = labels, [p.bbox for p in insts], hands
        own.update(frame, depth, per.process(frame, depth, t[0]), t[0])

    for _ in range(30):                                   # both raise -> refused
        step({RED: (WAIT, True), GREEN: (WAIT, True)})
    assert own.state == OwnerState.NO_OWNER and "ONE" in own.message
    assert len(checks) == 30                              # exactly one hand check per frame
    for _ in range(10):
        step({})
    for _ in range(25):                                   # only RED raises -> claimed within ~2.5 s
        step({RED: (WAIT, True)})
    assert own.state == OwnerState.CLAIMING
