"""Unit tests for the FollowMe modules (paper equations and transitions)."""

import math

import numpy as np
import pytest

from followme.calibration import CalibrationCollector, TargetModel, fit_target_model, identify
from followme.config import CalibrationConfig, FollowMeConfig, LocalizationConfig, NavigationConfig, \
    StateMachineConfig
from followme.gestures import FOLLOW, OTHER, WAIT, ConsecutiveFilter, GestureClassifier, landmark_features
from followme.localization import (CameraIntrinsics, TargetKalmanFilter, compute_goal, heading_angle,
                                   safety_radius, target_position, transform_point)
from followme.navigation import NavigationExecutor, RecordingNavigation
from followme.state_machine import Action, ActionKind, FollowMeStateMachine, Inputs, State
from followme.localization import Goal2D


# ---------------------------------------------------------------- calibration
def _clusters(rng, d=32, n=150):
    mu_t = rng.normal(0, 1, d)
    target = mu_t + rng.normal(0, 0.1, (n, d))
    other = mu_t + 1.0 + rng.normal(0, 0.1, (n, d))
    return target, other


def test_eq1_and_threshold_match_paper():
    rng = np.random.default_rng(1)
    target, _ = _clusters(rng)
    cfg = CalibrationConfig(sigma_floor_relative=0.0)
    model = fit_target_model(target, cfg, np.random.default_rng(7))
    order = np.random.default_rng(7).permutation(len(target))
    n_c = round(len(target) * 2 / 3)
    assert model.n_calibration == n_c and model.n_threshold == len(target) - n_c
    calib, thresh = target[order[:n_c]], target[order[n_c:]]
    np.testing.assert_allclose(model.mu, calib.mean(0))
    np.testing.assert_allclose(model.sigma, calib.std(0))
    x = thresh[0]
    d_manual = math.sqrt(np.mean(((x - calib.mean(0)) / calib.std(0)) ** 2))   # Eq. 1
    assert model.distances(x[None])[0] == pytest.approx(d_manual)
    d_all = model.distances(thresh)
    assert model.threshold == pytest.approx(d_all.mean() + 2 * d_all.std())     # lambda = mu_d + 2 sigma_d


def test_identify_least_distance_below_threshold():
    rng = np.random.default_rng(2)
    target, other = _clusters(rng)
    model = fit_target_model(target[:100], CalibrationConfig())
    frame = np.stack([other[120], target[120], other[121]])
    ident = identify(model, frame)
    assert ident.target_index == 1
    none = identify(model, np.stack([other[130], other[131]]))
    assert none.target_index is None and np.all(none.distances > model.threshold)
    masked = identify(model, frame, valid=np.array([True, False, True]))
    assert masked.target_index is None


def test_target_model_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    target, _ = _clusters(rng, d=8, n=30)
    model = fit_target_model(target, CalibrationConfig())
    model.meta["x"] = 1
    model.save(tmp_path / "t.npz")
    loaded = TargetModel.load(tmp_path / "t.npz")
    np.testing.assert_allclose(loaded.mu, model.mu)
    assert loaded.threshold == pytest.approx(model.threshold) and loaded.meta["x"] == 1


def test_collector_requires_single_person_and_spacing():
    cfg = CalibrationConfig(num_samples=6, min_interval_s=0.1, distance_bins_m=[1.0, 2.0], min_bins_covered=1)
    col = CalibrationCollector(cfg)
    f = np.ones((1, 4))
    assert not col.offer(np.ones((2, 4)), [1.0, 2.0], now=0.0)
    assert col.offer(f, [1.5], now=0.0)
    assert not col.offer(f, [1.5], now=0.05)
    for i in range(1, 6):
        assert col.offer(f + i, [0.5 + 0.4 * i], now=0.11 * i)
    assert col.done and col.rejected_multi == 1
    m = col.finish()
    assert m.n_calibration + m.n_threshold == 6 and m.n_threshold >= 3


# --------------------------------------------------------------- localization
INTR = CameraIntrinsics(600.0, 600.0, 320.0, 240.0, 640, 480)


def test_masked_centroid_deprojects_correctly():
    depth = np.full((480, 640), 5.0, np.float32)        # background far away
    mask = np.zeros((480, 640), bool)
    mask[140:340, 400:480] = True                        # person right of centre
    depth[mask] = 2.0
    cfg = LocalizationConfig(mask_erosion_px=3, pixel_stride=1)
    p = target_position(depth, mask, INTR, cfg)
    assert p[2] == pytest.approx(2.0)
    u = np.arange(403, 477).mean()
    assert p[0] == pytest.approx((u - 320) / 600 * 2.0, abs=1e-3)
    assert heading_angle(p) < 0                          # right of the camera -> negative (ROS)
    depth[:] = 0.0                                       # no valid depth
    assert target_position(depth, mask, INTR, cfg) is None


def test_kalman_filter_expiration():
    cfg = LocalizationConfig(expiration_time_s=3.0)
    kf = TargetKalmanFilter(cfg)
    assert kf.estimate(0.0) is None
    for k in range(30):
        t = k * 0.1
        kf.update(np.array([0.0, 0.0, 2.0 + 0.5 * t]), t)   # walking away at 0.5 m/s
    est = kf.estimate(3.0)
    assert est[2] == pytest.approx(3.5, abs=0.1)             # extrapolated with the velocity
    assert kf.estimate(2.9 + 2.9) is not None
    assert kf.estimate(2.9 + 3.01) is None                   # t_exp elapsed: stop integrating
    kf.update(np.array([1.0, 0.0, 1.0]), 10.0)               # re-initialised, not a huge innovation
    np.testing.assert_allclose(kf.estimate(10.0), [1.0, 0.0, 1.0])


def test_safety_radius_eq3():
    assert safety_radius(0.3, 3.0) == pytest.approx(1.26)    # paper reports 1.25 m


def test_goal_projection_and_transform():
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 0.5]
    p = transform_point(T, np.array([2.0, 0.0, 1.0]))
    np.testing.assert_allclose(p, [3.0, 2.0, 1.5])
    g = compute_goal(p, T[:3, 3])
    assert (g.x, g.y) == (3.0, 2.0) and g.yaw == pytest.approx(0.0) and g.distance == pytest.approx(2.0)
    g2 = compute_goal(p, T[:3, 3], standoff_m=0.5)
    assert g2.x == pytest.approx(2.5)


# ------------------------------------------------------------------- gestures
def _hand(rng, fist: bool):
    base = np.zeros((21, 3))
    for f in range(5):
        ang = -0.6 + 0.3 * f
        for j in range(4):
            r = (1.0 - 0.2 * j) if fist else (1.0 + j)      # fist: fingers fold back, MCPs stay
            base[1 + 4 * f + j] = [math.sin(ang) * r, -math.cos(ang) * r, 0.0]
    return base + rng.normal(0, 0.03, base.shape)


def test_landmark_features_invariance():
    rng = np.random.default_rng(0)
    h = _hand(rng, False)
    f1 = landmark_features(h * 3.0 + [100, 50, 7], "Right")
    f2 = landmark_features(h, "Right")
    np.testing.assert_allclose(f1, f2, atol=1e-5)
    mirrored = h.copy()
    mirrored[:, 0] *= -1
    np.testing.assert_allclose(landmark_features(mirrored, "Left"), f2, atol=1e-5)
    assert f1.shape == (63,)


def test_svm_separates_open_and_closed_hand(tmp_path):
    rng = np.random.default_rng(0)
    x, y = [], []
    for _ in range(80):
        x.append(landmark_features(_hand(rng, False)))
        y.append(WAIT)
        x.append(landmark_features(_hand(rng, True)))
        y.append(FOLLOW)
        x.append(rng.normal(0, 1, 63))
        y.append(OTHER)
    clf = GestureClassifier().fit(np.array(x[:180]), y[:180])
    acc = np.mean(clf.predict(np.array(x[180:])) == np.array(y[180:]))
    assert acc > 0.95
    clf.save(tmp_path / "g.joblib")
    assert GestureClassifier.load(tmp_path / "g.joblib").predict_one(x[0]) == WAIT


def test_consecutive_filter_xi():
    f = ConsecutiveFilter(5)
    outs = [f.update(WAIT) for _ in range(7)]
    assert outs == [None] * 4 + [WAIT, None, None]       # fires once, on the 5th
    assert f.update(OTHER) is None
    outs = [f.update(FOLLOW) for _ in range(4)] + [f.update(None)] + [f.update(FOLLOW) for _ in range(5)]
    assert outs[:5] == [None] * 5 and outs[-1] == FOLLOW  # a gap resets the count


# -------------------------------------------------------------- state machine
G = Goal2D(3.0, 0.0, 0.0, 3.0)


def test_state_machine_fig5_transitions():
    sm = FollowMeStateMachine(StateMachineConfig(initial_state="WAIT"))
    assert sm.step(Inputs(True, False)).kind == ActionKind.STOP and sm.state == State.WAIT
    a = sm.step(Inputs(True, False, FOLLOW, G))                       # gamma follow
    assert sm.state == State.FOLLOW and a.kind == ActionKind.GOAL
    assert sm.step(Inputs(True, True, None, G)).kind == ActionKind.STOP   # alpha & beta: stop, stay
    assert sm.state == State.FOLLOW
    sm.step(Inputs(False, True))                                       # beta & !alpha
    assert sm.state == State.STEADY
    sm.step(Inputs(True, False, None, G))                              # alpha
    assert sm.state == State.FOLLOW
    sm.note_heading(-0.4)
    a = sm.step(Inputs(False, False))                                  # !alpha & !beta
    assert sm.state == State.SEARCH and a.kind == ActionKind.ROTATE and a.direction == -1
    sm.step(Inputs(True, False, None, G))                              # alpha during search
    assert sm.state == State.FOLLOW
    sm.step(Inputs(False, False))
    sm.step(Inputs(False, False, search_angle_done=2 * math.pi))       # full turn
    assert sm.state == State.STEADY
    sm.step(Inputs(True, False, None, G))
    sm.step(Inputs(True, False, WAIT, G))                              # gamma wait
    assert sm.state == State.WAIT
    assert sm.step(Inputs(True, False, None, G)).kind == ActionKind.STOP


def test_steady_can_require_gesture():
    sm = FollowMeStateMachine(StateMachineConfig(initial_state="STEADY", steady_requires_follow_gesture=True))
    sm.step(Inputs(True, False, None, G))
    assert sm.state == State.STEADY
    sm.step(Inputs(True, False, FOLLOW, G))
    assert sm.state == State.FOLLOW


# ----------------------------------------------------------------- navigation
def test_executor_goal_throttle_stop_and_search_angle():
    nav = RecordingNavigation()
    ex = NavigationExecutor(NavigationConfig(goal_update_distance_m=0.2, goal_update_period_s=1.0), nav)
    ex.execute(Action(ActionKind.GOAL, Goal2D(1.0, 0.0, 0.0, 1.0)), 0.0)
    ex.execute(Action(ActionKind.GOAL, Goal2D(1.05, 0.0, 0.0, 1.0)), 0.1)      # too small, too soon
    ex.execute(Action(ActionKind.GOAL, Goal2D(1.5, 0.0, 0.0, 1.0)), 0.2)
    assert [c[0] for c in nav.calls] == ["goal", "goal"]
    ex.execute(Action(ActionKind.STOP), 0.3)
    assert nav.calls[-2:] == [("cancel",), ("vel", 0.0, 0.0)]
    for k in range(5):
        ex.execute(Action(ActionKind.ROTATE, direction=1), 1.0 + k)
    assert ex.search_angle == pytest.approx(4 * 0.5)
    ex2 = NavigationExecutor(NavigationConfig(), RecordingNavigation())
    for k, yaw in enumerate([3.0, -3.0, -2.5]):                                  # odometry, wraps at pi
        ex2.execute(Action(ActionKind.ROTATE, direction=1), float(k), robot_yaw=yaw)
    assert ex2.search_angle == pytest.approx((2 * math.pi - 6.0) + 0.5)


def test_config_yaml_matches_defaults():
    from pathlib import Path
    cfg = FollowMeConfig.from_yaml(Path(__file__).resolve().parents[1] / "config" / "followme.yaml")
    assert cfg.to_dict() == FollowMeConfig().to_dict()
    with pytest.raises(KeyError):
        FollowMeConfig.from_dict({"reid": {"typo": 1}})


# ----------------------------------------------------------------- evaluation
def test_reid_protocol_and_table1_metrics():
    from followme.evaluation import REID_LABELS, ImageInstances, classification_report, evaluate_reid_subject
    rng = np.random.default_rng(5)
    d = 16
    centres = {s: rng.normal(0, 1, d) for s in "ABC"}
    feat = lambda s: centres[s] + rng.normal(0, 0.08, d)          # noqa: E731
    boxes = np.array([[0, 0, 10, 20], [20, 0, 30, 20], [40, 0, 50, 20]], np.float32)
    calib = np.stack([feat("A") for _ in range(60)])
    present = [ImageInstances(np.stack([feat("B"), feat("A"), feat("C")]), boxes, target_box=boxes[1])
               for _ in range(20)]
    absent = [ImageInstances(np.stack([feat("B"), feat("C")]), boxes[:2]) for _ in range(20)]
    r = evaluate_reid_subject("A", calib, present, absent, CalibrationConfig())
    assert r.cm.sum() == 40 and r.wrong_person == 0
    assert r.cm[1, 0] == 0                                          # no false positives
    assert r.cm[0, 0] >= 15                                         # most targets found
    rep = classification_report(r.cm, REID_LABELS)
    assert rep["Target"]["precision"] == 1.0 and 0.0 < rep["accuracy"] <= 1.0
    assert len(r.target_distances) == r.model.n_threshold and len(r.distractor_distances) == 80


# ------------------------------------------------ gallery metric (extension)
def _unitv(v):
    return v / np.linalg.norm(v)


def _cluster(rng, centre, n, noise=0.05):
    return np.stack([_unitv(centre + rng.normal(0, noise, centre.shape)) for _ in range(n)])


def test_gallery_handles_multiple_views_where_prototype_fails():
    from followme.calibration import IdentityMemory
    from followme.config import IdentityConfig
    rng = np.random.default_rng(0)
    d = 64
    front, back = np.zeros(d), np.zeros(d)
    front[0], back[1] = 1.0, 1.0
    calib = np.vstack([_cluster(rng, front, 60, 0.03), _cluster(rng, back, 60, 0.03)])
    between = _unitv(front + back)                          # an impostor "in between" both views
    model = fit_target_model(calib, CalibrationConfig(metric="gallery"))
    mem = IdentityMemory(IdentityConfig())
    for probe in (front, back):
        assert identify(model, _cluster(rng, probe, 1, 0.03), memory=mem).target_index == 0
    assert identify(model, between[None], memory=mem).target_index is None
    model.metric = "cosine"                                 # a single average cannot tell
    assert identify(model, between[None]).target_index == 0


def test_negative_bank_rejects_a_known_stranger():
    """A look-alike who passes the gallery threshold on ~30 % of frames is almost always
    rejected once seen next to the target, while the target keeps being accepted."""
    from followme.calibration import IdentityMemory
    from followme.config import IdentityConfig
    imp_before, imp_after, me_after = [], [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        me = _unitv(rng.normal(0, 1, 64))
        other = _unitv(me + 0.3 * _unitv(rng.normal(0, 1, 64)))
        model = fit_target_model(_cluster(rng, me, 120), CalibrationConfig(metric="gallery"))
        mem = IdentityMemory(IdentityConfig(online_update=False))
        imp_before += [identify(model, _cluster(rng, other, 1), memory=mem).target_index == 0 for _ in range(50)]
        boxes = [np.array([0, 0, 10, 30.]), np.array([100, 0, 110, 30.])]
        for _ in range(30):                                  # seen together, the target is chosen
            feats = np.vstack([_cluster(rng, me, 1), _cluster(rng, other, 1)])
            r = identify(model, feats, memory=mem)
            if r.target_index == 0:
                mem.observe(model, feats, boxes, np.array([True, True]), 0, float(r.distances[0]), 0.0)
        assert mem.counters["negatives_added"] > 0
        imp_after += [identify(model, _cluster(rng, other, 1), memory=mem).target_index == 0 for _ in range(50)]
        me_after += [identify(model, _cluster(rng, me, 1), memory=mem).target_index == 0 for _ in range(50)]
    assert np.mean(imp_before) > 0.2
    assert np.mean(imp_after) < 0.05
    assert np.mean(me_after) > 0.8


def test_online_views_and_persistence(tmp_path):
    from followme.calibration import IdentityMemory, TargetModel
    from followme.config import IdentityConfig
    rng = np.random.default_rng(2)
    me = _unitv(rng.normal(0, 1, 32))
    model = fit_target_model(_cluster(rng, me, 60, 0.05), CalibrationConfig(metric="gallery"))
    mem = IdentityMemory(IdentityConfig(online_interval_s=1.0, online_duplicate_sim=0.999))
    box = [np.array([0, 0, 10, 30.])]
    for t in (0.0, 0.5, 1.2, 2.5):
        f = _cluster(rng, me, 1, 0.02)
        mem.observe(model, f, box, np.array([True]), 0, float(model.gallery_distances(f)[0]), t)
    assert mem.counters["views_added"] == 3                 # t = 0.5 is too soon
    model.save(tmp_path / "g.npz")
    loaded = TargetModel.load(tmp_path / "g.npz")
    assert loaded.active_metric == "gallery" and loaded.gal_threshold == pytest.approx(model.gal_threshold)
    np.testing.assert_allclose(loaded.gallery, model.gallery, atol=1e-6)


# ------------------------------------------------------- direct cmd_vel control
def test_direct_follow_controller():
    from followme.direct_control import DirectFollowController
    cfg = NavigationConfig(max_velocity_mps=0.3, accel_mps2=0.4, decel_mps2=1.0)
    c = DirectFollowController(cfg, safety_radius=1.26)
    assert c.follow_distance == pytest.approx(1.56)
    vs = [c.command(Goal2D(3.0, 0.0, 0.0, 3.0), 0.1 * k)[0] for k in range(1, 20)]
    assert vs[0] == pytest.approx(0.04) and max(vs) == pytest.approx(0.3)     # ramps to v_max
    v, w = c.command(Goal2D(1.0, 1.0, math.pi / 4, 1.41), 2.1)                  # target left, close
    assert w > 0 and v < 0.3
    c.reset()
    v, w = c.command(Goal2D(1.0, -0.2, -0.2, 1.2), 0.0)                        # inside follow distance
    assert v == 0.0 and w < 0
    v, w = c.command(Goal2D(0.0, 3.0, math.pi / 2, 3.0), 0.1)                  # 90 deg: turn, no speed
    assert v == pytest.approx(0.0, abs=1e-9) and w == pytest.approx(0.8)


def test_stop_and_look_search():
    nav = RecordingNavigation()
    ex = NavigationExecutor(NavigationConfig(search_angular_speed=0.3, search_move_s=1.0,
                                             search_pause_s=0.5), nav)
    moving = []
    for k in range(31):                                   # 3 s at 10 Hz
        ex.execute(Action(ActionKind.ROTATE, direction=1), 0.1 * k)
        moving.append(nav.calls[-1][2] != 0.0)
    assert moving[:10] == [True] * 10 and not any(moving[11:15]) and moving[16]
    assert ex.search_angle == pytest.approx(0.3 * 2.0, abs=0.07)   # only moving time counts


def test_adaptive_views_only_when_owner_is_alone():
    from followme.calibration import IdentityMemory
    from followme.config import IdentityConfig
    rng = np.random.default_rng(4)
    me = _unitv(rng.normal(0, 1, 32))
    model = fit_target_model(_cluster(rng, me, 60, 0.05), CalibrationConfig(metric="gallery"))
    cfg = IdentityConfig(online_from_tracking=True, tracking_threshold_factor=1.8, online_interval_s=0.0)
    mem = IdentityMemory(cfg)
    d = 1.4 * model.gal_threshold                     # "new room": too far for normal online learning
    f = _unitv(me + 0.4 * _unitv(rng.normal(0, 1, 32)))[None]   # a genuinely new view
    b = [np.array([0, 0, 10, 30.]), np.array([100, 0, 110, 30.])]
    mem.observe(model, np.vstack([f, f]), b, np.array([True, True]), 0, d, 0.0)   # someone else in view
    assert mem.counters["views_added"] == 0
    mem.observe(model, f, b[:1], np.array([True]), 0, d, 1.0)                       # alone
    assert mem.counters["views_added"] == 1
    mem.observe(model, f, b[:1], np.array([True]), 0, 1.9 * model.gal_threshold, 2.0)  # beyond 1.8x
    assert mem.counters["views_added"] == 1


# ---------------------------------------------------------------- mirror_x
def test_mirror_x_flips_target_position_x_only():
    depth = np.full((480, 640), 5.0, np.float32)
    mask = np.zeros((480, 640), bool)
    mask[140:340, 400:480] = True             # person on the right side of the image
    depth[mask] = 2.0
    cfg_plain = LocalizationConfig(mask_erosion_px=3, pixel_stride=1)
    cfg_mirror = LocalizationConfig(mask_erosion_px=3, pixel_stride=1, mirror_x=True)
    from followme.localization import target_position
    p = target_position(depth, mask, INTR, cfg_plain)
    pm = target_position(depth, mask, INTR, cfg_mirror)
    assert p[0] > 0                                    # right of camera centre -> positive x normally
    assert pm[0] == pytest.approx(-p[0])                # mirrored: sign flipped
    assert pm[1] == pytest.approx(p[1]) and pm[2] == pytest.approx(p[2])   # y, z unaffected
    # and heading_angle now says "the opposite side" once mirrored
    assert heading_angle(p) < 0
    assert heading_angle(pm) > 0


def test_mirror_x_flips_bbox_fallback_too():
    from followme.localization import position_from_bbox
    bbox = np.array([420.0, 100.0, 480.0, 400.0])       # box on the right
    p = position_from_bbox(bbox, INTR, 1.7, mirror_x=False)
    pm = position_from_bbox(bbox, INTR, 1.7, mirror_x=True)
    assert pm[0] == pytest.approx(-p[0])


def test_deploy_config_is_complete_and_valid():
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "config" / "followme_deploy.yaml"
    c = FollowMeConfig.from_yaml(p)
    assert c.navigation.max_velocity_mps == 0.05
    assert c.navigation.safety_radius_m == 0.5 and c.navigation.follow_distance_m == 0.6
    assert c.navigation.search_angular_speed == 0.0 and c.navigation.search_linear_speed == 0.0
    assert c.calibration.metric == "gallery" and c.identity.online_from_tracking
    assert c.ownership.enabled and c.ownership.hands_per_frame == 1
    assert c.localization.gate_m == 1.5
