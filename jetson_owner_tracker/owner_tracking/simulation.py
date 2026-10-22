"""Deterministic synthetic scenes for testing identity preservation.

People are rendered as coloured bodies with depth ordering, so crossings
produce genuine occlusion and contaminated crops. The detector is simulated
from the rendered visibility (a person is detected only if enough of them is
visible), with bbox jitter and confidence noise. Ground-truth identities are
exposed separately and never shown to the tracker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

import numpy as np

from owner_tracking.datatypes import Detection

BGR = tuple[int, int, int]
SKIN: BGR = (140, 170, 210)


@dataclass(frozen=True)
class MotionSegment:
    """Linear motion between two frames (inclusive)."""

    start_frame: int
    end_frame: int
    start_center: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)


@dataclass
class SyntheticPerson:
    person_id: int
    segments: list[MotionSegment]
    top_color: BGR
    bottom_color: BGR
    size: tuple[float, float] = (60.0, 160.0)
    detection_confidence: float = 0.90
    depth: int = 0          # larger = closer to the camera (drawn on top)

    def bbox_at(self, t: int) -> Optional[np.ndarray]:
        for seg in self.segments:
            if seg.start_frame <= t <= seg.end_frame:
                dt = t - seg.start_frame
                cx = seg.start_center[0] + seg.velocity[0] * dt
                cy = seg.start_center[1] + seg.velocity[1] * dt
                w, h = self.size
                return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)
        return None


@dataclass
class Occluder:
    bbox: tuple[int, int, int, int]
    color: BGR = (70, 110, 150)


@dataclass
class ScenarioFrame:
    index: int
    frame: np.ndarray
    detections: list[Detection]
    ground_truth: dict[int, np.ndarray]     # visible people only


@dataclass
class SyntheticScene:
    persons: list[SyntheticPerson]
    num_frames: int
    occluders: list[Occluder] = field(default_factory=list)
    width: int = 640
    height: int = 480
    jitter_px: float = 1.5
    confidence_noise: float = 0.03
    min_visible_fraction: float = 0.35
    miss_rate: float = 0.0            # probability a visible person is not detected
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0, 6, (self.height, self.width, 1))
        self._background = np.clip(90 + noise, 0, 255).astype(np.uint8).repeat(3, axis=2)

    def _clip(self, box: np.ndarray) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = np.round(box).astype(int)
        return (max(0, x1), max(0, y1), min(self.width, x2), min(self.height, y2))

    def render(self, t: int) -> ScenarioFrame:
        frame = self._background.copy()
        idmap = np.full((self.height, self.width), -1, dtype=np.int32)
        drawn: dict[int, int] = {}
        boxes: dict[int, np.ndarray] = {}
        for person in sorted(self.persons, key=lambda p: p.depth):
            box = person.bbox_at(t)
            if box is None:
                continue
            boxes[person.person_id] = box
            x1, y1, x2, y2 = self._clip(box)
            if x2 <= x1 or y2 <= y1:
                continue
            h = y2 - y1
            head_y2 = y1 + int(0.15 * h)
            torso_y2 = y1 + int(0.55 * h)
            hx1 = x1 + int(0.3 * (x2 - x1))
            hx2 = x2 - int(0.3 * (x2 - x1))
            before = int(np.count_nonzero(idmap == person.person_id))
            frame[y1:head_y2, hx1:hx2] = SKIN
            frame[head_y2:torso_y2, x1:x2] = person.top_color
            frame[torso_y2:y2, x1:x2] = person.bottom_color
            idmap[y1:head_y2, hx1:hx2] = person.person_id
            idmap[head_y2:y2, x1:x2] = person.person_id
            drawn[person.person_id] = (head_y2 - y1) * (hx2 - hx1) + (y2 - head_y2) * (x2 - x1) + before
        for occ in self.occluders:
            x1, y1, x2, y2 = occ.bbox
            frame[y1:y2, x1:x2] = occ.color
            idmap[y1:y2, x1:x2] = -2
        rng = np.random.default_rng(self.seed * 100_003 + t)
        detections: list[Detection] = []
        ground_truth: dict[int, np.ndarray] = {}
        for person in self.persons:
            pid = person.person_id
            if pid not in boxes or drawn.get(pid, 0) <= 0:
                continue
            visible = np.count_nonzero(idmap == pid) / drawn[pid]
            if visible < self.min_visible_fraction:
                continue
            ground_truth[pid] = boxes[pid]
            if rng.random() < self.miss_rate:
                continue
            noisy = boxes[pid] + rng.normal(0, self.jitter_px, 4).astype(np.float32)
            conf = float(np.clip(person.detection_confidence * (0.7 + 0.3 * visible)
                                 + rng.normal(0, self.confidence_noise), 0.05, 0.99))
            detections.append(Detection(noisy, conf, source_id=pid))
        return ScenarioFrame(t, frame, detections, ground_truth)

    def __iter__(self) -> Iterator[ScenarioFrame]:
        for t in range(self.num_frames):
            yield self.render(t)


@dataclass
class Scenario:
    name: str
    description: str
    scene: SyntheticScene
    owner_id: int
    expect_recovery: bool
    recovery_after_frame: int = 0


RED: BGR = (40, 40, 200)
NAVY: BGR = (110, 40, 20)
GREEN: BGR = (40, 170, 40)
KHAKI: BGR = (110, 170, 190)
PURPLE: BGR = (150, 40, 130)
TEAL: BGR = (150, 150, 20)
Y = 300.0


def _walker(pid: int, x: float, vx: float, top: BGR, bottom: BGR, frames: int,
            depth: int = 0, conf: float = 0.9, y: float = Y) -> SyntheticPerson:
    return SyntheticPerson(pid, [MotionSegment(0, frames - 1, (x, y), (vx, 0.0))], top, bottom,
                           detection_confidence=conf, depth=depth)


def build_scenarios(seed: int = 0, jitter_px: float = 1.5, miss_rate: float = 0.0) -> dict[str, Scenario]:
    """The standard identity-preservation test suite.

    ``jitter_px`` and ``miss_rate`` control detector noise for stress tests.
    """
    s: dict[str, Scenario] = {}
    n = 90
    s["crossing_distinct"] = Scenario(
        "crossing_distinct", "Owner walks behind a differently dressed person crossing paths.",
        SyntheticScene([_walker(1, 120, 5, RED, NAVY, n), _walker(2, 520, -5, GREEN, KHAKI, n, depth=1)],
                       n, seed=seed), owner_id=1, expect_recovery=True, recovery_after_frame=50)
    s["crossing_similar"] = Scenario(
        "crossing_similar", "Two identically dressed people cross; owner is occluded.",
        SyntheticScene([_walker(1, 120, 5, RED, NAVY, n), _walker(2, 520, -5, RED, NAVY, n, depth=1)],
                       n, seed=seed), owner_id=1, expect_recovery=False)
    s["crossing_similar_owner_front"] = Scenario(
        "crossing_similar_owner_front", "Identically dressed people cross; owner in front.",
        SyntheticScene([_walker(1, 120, 5, RED, NAVY, n, depth=1), _walker(2, 520, -5, RED, NAVY, n)],
                       n, seed=seed), owner_id=1, expect_recovery=False)
    m = 120
    s["side_by_side_similar"] = Scenario(
        "side_by_side_similar", "Similar-looking people walk together, converge, partially overlap, diverge.",
        SyntheticScene([
            _walker(1, 120, 2, RED, NAVY, m),
            SyntheticPerson(2, [MotionSegment(0, 39, (195, Y), (2, 0)),
                                MotionSegment(40, 69, (275, Y), (1, 0)),
                                MotionSegment(70, m - 1, (305, Y), (3, 0))], RED, NAVY, depth=1),
        ], m, seed=seed), owner_id=1, expect_recovery=False)
    k = 150
    s["occlusion_with_intruders"] = Scenario(
        "occlusion_with_intruders",
        "Owner passes behind a pillar; a high-confidence stranger and a look-alike appear meanwhile.",
        SyntheticScene([
            _walker(1, 120, 3, RED, NAVY, k),
            SyntheticPerson(2, [MotionSegment(40, k - 1, (560, Y), (-0.3, 0))], GREEN, KHAKI,
                            detection_confidence=0.99),
            SyntheticPerson(3, [MotionSegment(50, 95, (80, 330), (0.2, 0))], RED, NAVY,
                            detection_confidence=0.99),
        ], k, occluders=[Occluder((290, 160, 390, 440))], seed=seed),
        owner_id=1, expect_recovery=True, recovery_after_frame=100)
    s["high_confidence_intruder"] = Scenario(
        "high_confidence_intruder", "Low-confidence owner; a 0.99-confidence stranger passes in front.",
        SyntheticScene([_walker(1, 150, 3, PURPLE, NAVY, n, conf=0.55),
                        _walker(2, 520, -4, TEAL, KHAKI, n, depth=1, conf=0.99, y=Y + 15)],
                       n, seed=seed), owner_id=1, expect_recovery=True, recovery_after_frame=60)
    L = 160
    s["long_term_reid"] = Scenario(
        "long_term_reid", "Owner leaves the scene for 70 frames and returns elsewhere; a stranger stays.",
        SyntheticScene([
            SyntheticPerson(1, [MotionSegment(0, 29, (150, Y), (2, 0)),
                                MotionSegment(100, L - 1, (480, Y), (-2, 0))], RED, NAVY),
            SyntheticPerson(2, [MotionSegment(0, L - 1, (590, Y), (0, 0))], GREEN, KHAKI),
        ], L, seed=seed), owner_id=1, expect_recovery=True, recovery_after_frame=100)
    for scenario in s.values():
        scenario.scene.jitter_px = jitter_px
        scenario.scene.miss_rate = miss_rate
    return s


def run_scenario(scenario: Scenario, tracker) -> list:
    """Run ``tracker`` over a scenario.

    The simulated operator clicks the owner's upper body on the first frame in
    which the owner is detected (like a human re-clicking after a miss).
    Returns a list of ``(ScenarioFrame, FrameResult)`` pairs.
    """
    results = []
    for sf in scenario.scene:
        result = tracker.update(sf.frame, sf.detections)
        if not tracker.has_owner and any(d.source_id == scenario.owner_id for d in sf.detections):
            box = sf.ground_truth[scenario.owner_id]
            click = ((box[0] + box[2]) / 2, box[1] + 0.4 * (box[3] - box[1]))
            ok, msg = tracker.select_owner(sf.frame, click, sf.detections)
            if not ok:
                raise RuntimeError(f"Owner selection failed: {msg}")
        results.append((sf, result))
    return results
