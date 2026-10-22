"""End-to-end FollowMe test on a synthetic RGB-D scene (no GPU, no models).

Segmentation is replaced by ground-truth masks and Re-ID by colour
histograms, so this validates the *pipeline logic* (calibration, Eq. 1
identification, masked-depth localisation, KF expiration, state machine and
navigation calls), not the accuracy of the deep models.
"""

import math

import numpy as np
import pytest

from followme.config import FollowMeConfig
from followme.gestures import FOLLOW, WAIT
from followme.localization import CameraIntrinsics
from followme.navigation import RecordingNavigation
from followme.pipeline import FollowMeController, FollowMePerception, Mode
from followme.reid import OnnxReID, build_feature_extractor
from followme.segmentation import PersonInstance, StaticSegmenter
from followme.state_machine import ActionKind, State

INTR = CameraIntrinsics(600.0, 600.0, 320.0, 240.0, 640, 480)
TARGET = ((40, 40, 200), (160, 60, 20))      # BGR: red shirt, blue trousers
OTHER = ((40, 180, 40), (40, 200, 220))      # green shirt, yellow trousers


def render(people, rng):
    """people: list of (colours, x_m, z_m). Returns frame, depth, instances, colours per instance."""
    frame = rng.integers(90, 130, (480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), 7.0, np.float32)
    insts, labels = [], []
    for (top, bottom), x_m, z_m in sorted(people, key=lambda p: -p[2]):
        h = 600 * 1.7 / z_m
        w = h * 0.3
        u = 320 + 600 * x_m / z_m
        x1, x2 = int(u - w / 2), int(u + w / 2)
        y1, y2 = int(240 - h / 2), int(240 + h / 2)
        x1c, x2c, y1c, y2c = max(0, x1), min(640, x2), max(0, y1), min(480, y2)
        mask = np.zeros((480, 640), bool)
        mask[y1c:y2c, x1c:x2c] = True
        gain = rng.uniform(0.8, 1.2)
        mid = (y1c + y2c) // 2
        frame[y1c:mid, x1c:x2c] = np.clip(np.array(top) * gain + rng.normal(0, 6, (mid - y1c, x2c - x1c, 3)), 0, 255)
        frame[mid:y2c, x1c:x2c] = np.clip(np.array(bottom) * gain + rng.normal(0, 6, (y2c - mid, x2c - x1c, 3)),
                                          0, 255)
        depth[mask] = z_m
        for other in insts:                       # occlusion: nearer person hides farther one
            other.mask &= ~mask
        insts.append(PersonInstance([x1c, y1c, x2c, y2c], mask, 0.9))
        labels.append((top, bottom))
    return frame, depth, insts, labels


@pytest.fixture(params=["eq1", "gallery"])
def setup(request):
    cfg = FollowMeConfig()
    cfg.calibration.metric = request.param
    cfg.reid.backend = "histogram"
    cfg.gestures.enabled = False
    cfg.calibration.num_samples = 60
    seg = StaticSegmenter()
    perception = FollowMePerception(cfg, seg, build_feature_extractor(cfg.reid, "cpu"), INTR)
    return cfg, seg, perception


def run(perception, seg, people, rng, t):
    frame, depth, insts, labels = render(people, rng)
    seg.next = insts
    return perception.process(frame, depth, t), labels


def test_calibrate_identify_localize_and_follow(setup):
    cfg, seg, perception = setup
    rng = np.random.default_rng(0)
    perception.start_calibration()
    t = 0.0
    # a second person in view is refused during calibration
    run(perception, seg, [(TARGET, 0.0, 2.0), (OTHER, 1.0, 3.0)], rng, t)
    assert perception.collector.rejected_multi == 1
    while perception.mode == Mode.CALIBRATING:
        t += 0.07
        z = 1.5 + 2.5 * (0.5 + 0.5 * math.sin(t))          # walk near / far
        run(perception, seg, [(TARGET, 0.3 * math.cos(3 * t), z)], rng, t)
    assert perception.model is not None and perception.model.meta["distance_bins_covered"] >= 2

    nav = RecordingNavigation()
    ctl = FollowMeController(cfg, nav)
    assert ctl.state == State.WAIT
    correct = wrong = 0
    for k in range(40):                                  # target + distractor, crossing in front
        t += 0.1
        people = [(TARGET, -0.5 + 0.02 * k, 3.0), (OTHER, 0.8 - 0.02 * k, 2.5 + 0.02 * k)]
        r, labels = run(perception, seg, people, rng, t)
        if r.target_index is not None:
            if labels[r.target_index] == TARGET:
                correct += 1
            else:
                wrong += 1
        if k == 5:
            ctl.command(FOLLOW, t)                       # the target's Follow gesture
        ctl.step(r)
    assert wrong == 0                                    # never the distractor (paper: FP are the costly error)
    assert correct >= 30                                 # misses only around the occlusion (KF bridges them)
    assert ctl.state == State.FOLLOW
    if cfg.calibration.metric == "gallery":
        assert perception.memory.counters["negatives_added"] > 0
    assert r.p_filt[2] == pytest.approx(3.0, abs=0.05)
    goals = [c for c in nav.calls if c[0] == "goal"]
    assert goals and goals[-1][1].x == pytest.approx(3.0, abs=0.1)      # base frame: x forward

    # only the distractor remains: never re-identified; KF expires after t_exp -> SEARCH
    t_lost = t
    while t - t_lost < cfg.localization.expiration_time_s + 0.5:
        t += 0.1
        r, _ = run(perception, seg, [(OTHER, 0.2, 2.5)], rng, t)
        assert r.target_index is None
        ctl.step(r)
    assert ctl.state == State.SEARCH
    assert nav.calls[-1][0] == "vel" and nav.calls[-1][2] != 0.0
    while ctl.state == State.SEARCH and t - t_lost < 30:
        t += 0.1
        r, _ = run(perception, seg, [(OTHER, 0.2, 2.5)], rng, t)
        ctl.step(r)
    assert ctl.state == State.STEADY                    # one full turn without alpha
    assert nav.calls[-1] == ("vel", 0.0, 0.0)

    # target comes back close: alpha -> FOLLOW, but inside the safety circle -> stop
    t += 0.1
    r, _ = run(perception, seg, [(TARGET, 0.0, 1.0)], rng, t)
    a = ctl.step(r)
    assert ctl.state == State.FOLLOW and a.kind == ActionKind.STOP
    ctl.command(WAIT, t)
    assert ctl.state == State.WAIT


def test_onnx_backend(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper
    from followme.config import ReIDConfig

    node1 = helper.make_node("GlobalAveragePool", ["images"], ["pooled"])
    node2 = helper.make_node("Flatten", ["pooled"], ["features"], axis=1)
    graph = helper.make_graph([node1, node2], "tiny",
                              [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, 256, 128])],
                              [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch", 3])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "tiny.onnx"
    onnx.save(model, path)
    ext = OnnxReID(ReIDConfig(backend="onnx", model_path=str(path), batch_size=2), "cpu")
    crops = [np.full((100, 40, 3), (255, 0, 0), np.uint8), np.full((50, 30, 3), (0, 0, 255), np.uint8),
             np.full((60, 20, 3), (0, 255, 0), np.uint8)]
    f = ext.extract(crops)
    assert f.shape == (3, 3) and ext.feature_dim == 3
    np.testing.assert_allclose(np.linalg.norm(f, axis=1), 1.0, rtol=1e-5)
    assert np.argmax(f[0]) == 2 and np.argmax(f[1]) == 0     # BGR crop -> RGB features


def test_mmt_network_shapes():
    torch = pytest.importorskip("torch")
    from followme.reid import build_mmt_ibn_resnet50

    net = build_mmt_ibn_resnet50(256).eval()
    keys = net.state_dict().keys()
    assert "base.4.0.bn1.IN.weight" in keys and "base.7.0.bn1.weight" in keys and "feat.weight" in keys
    with torch.no_grad():
        out = net(torch.zeros(2, 3, 256, 128))
    assert out.shape == (2, 256)


def test_tracking_hysteresis_holds_target_but_not_strangers():
    cfg = FollowMeConfig()
    cfg.reid.backend = "histogram"
    cfg.gestures.enabled = False
    cfg.calibration.num_samples = 60
    cfg.calibration.metric = "gallery"
    cfg.localization.gate_m = 1.0
    cfg.identity.tracking_threshold_factor = 40.0   # histogram distances are tiny and noisy
    seg = StaticSegmenter()
    perception = FollowMePerception(cfg, seg, build_feature_extractor(cfg.reid, "cpu"), INTR)
    rng = np.random.default_rng(3)
    perception.start_calibration()
    t = 0.0
    while perception.mode == Mode.CALIBRATING:
        t += 0.07
        run(perception, seg, [(TARGET, 0.2 * math.cos(3 * t), 1.5 + 2.5 * (0.5 + 0.5 * math.sin(t)))], rng, t)
    t += 0.1
    r, labels = run(perception, seg, [(TARGET, 0.0, 2.5)], rng, t)
    assert labels[r.target_index] == TARGET                      # tracked (KF active)
    perception.model.gal_threshold = 1.5e-4                      # strictly every frame now fails
    held = 0
    for k in range(10):
        t += 0.1
        r, labels = run(perception, seg, [(TARGET, 0.02 * k, 2.5)], rng, t)
        held += r.target_index is not None and labels[r.target_index] == TARGET
    assert held >= 8 and perception.relaxed_accepts >= 8
    for _ in range(40):                                          # target gone, stranger elsewhere
        t += 0.1
        r, _ = run(perception, seg, [(OTHER, -1.0, 4.0)], rng, t)
        assert r.target_index is None
