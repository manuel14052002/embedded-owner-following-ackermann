"""Lidar owner fusion core: clustering, camera association, lidar-only tracking."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2" / "followme_ros"))
from followme_ros.lidar_owner_fusion import OwnerLidarTracker, cluster_points, scan_to_points  # noqa: E402


def make_scan(people, wall=4.0, n=720):
    """Polar scan with a circular wall and each person as two legs (r=6 cm, 20 cm apart)."""
    a = -math.pi + 2 * math.pi * np.arange(n) / n
    r = np.full(n, wall)
    for (px, py) in people:
        for dx in (-0.1, 0.1):
            lx, ly = px + dx * -math.sin(math.atan2(py, px)), py + dx * math.cos(math.atan2(py, px))
            d = math.hypot(lx, ly)
            ang = math.atan2(ly, lx)
            half = math.atan2(0.06, d)
            m = np.abs((a - ang + math.pi) % (2 * math.pi) - math.pi) < half
            r[m] = np.minimum(r[m], d - 0.03)
    return r, -math.pi, 2 * math.pi / n


def clusters_for(people):
    r, amin, inc = make_scan(people)
    return cluster_points(scan_to_points(r, amin, inc, 0.1, 12.0))


def test_legs_merge_into_one_person_cluster():
    cl = clusters_for([(2.0, 0.0), (1.5, 1.5)])
    for person in ((2.0, 0.0), (1.5, 1.5)):        # each person = ONE cluster (both legs merged)
        near = [c for c in cl if np.linalg.norm(c.centroid - person) < 0.3]
        assert len(near) == 1 and near[0].width > 0.2


def test_camera_association_then_lidar_only_tracking_around_a_corner():
    tr = OwnerLidarTracker()
    t = 0.0
    owner, other = np.array([2.0, 0.0]), np.array([1.5, -1.5])
    fused = tr.update(clusters_for([owner, other]), t, camera_xy=owner + [0.1, 0.05], camera_t=t)
    assert np.linalg.norm(fused - owner) < 0.15 and tr.source == "camera+lidar"
    # owner walks out of the camera view (to the left and behind the robot); camera silent
    for k in range(1, 60):
        t += 0.1
        ang = min(k * 0.05, math.pi * 0.8)
        owner = np.array([2.0 * math.cos(ang), 2.0 * math.sin(ang)])
        fused = tr.update(clusters_for([owner, other]), t)
        assert fused is not None and np.linalg.norm(fused - owner) < 0.2, k
    assert tr.source == "lidar"
    assert np.linalg.norm(fused - other) > 1.0                  # never switched to the other person


def test_track_dropped_when_owner_disappears():
    tr = OwnerLidarTracker(lost_s=0.5)
    tr.update(clusters_for([(2.0, 0.0)]), 0.0, camera_xy=np.array([2.0, 0.0]), camera_t=0.0)
    for k in range(1, 10):
        out = tr.update(clusters_for([]), 0.1 * k)
    assert out is None and tr.pos is None


def test_no_association_when_camera_point_far_from_any_cluster():
    tr = OwnerLidarTracker(assoc_gate=0.6)
    out = tr.update(clusters_for([(2.0, 0.0)]), 0.0, camera_xy=np.array([0.0, 3.0]), camera_t=0.0)
    assert out is None
