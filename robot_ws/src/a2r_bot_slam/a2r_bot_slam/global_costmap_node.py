#!/usr/bin/env python3
"""
global_costmap_node.py
======================
Inflates /map into /global_costmap.

The map SLAM produces describes the world.  It says nothing about the robot,
and in particular it does not know the car is 31 cm wide.  Planning a point
robot across that map produces routes through 20 cm gaps, which is a very
convincing-looking plan and an immediate collision.

There are two ways to fix that.  Check the full footprint at every step of the
search — correct, and roughly two orders of magnitude slower.  Or grow every
obstacle by the robot's inscribed radius once, then plan a point across the
grown map.  The second is what everyone does, and it is exact for a circular
footprint; for the car it is slightly conservative at the corners, which is
the direction you want to be wrong in.

The inflation has two zones:

  **Lethal** (within `robot_radius_m`)  — the robot's centre cannot be here,
  because its body would overlap an obstacle.  Marked 100.

  **Inflated** (out to `inflation_radius_m`) — passable, but increasingly
  expensive.  This is what makes a planner drive down the middle of a corridor
  rather than shaving the wall.  The cost decays exponentially so there is no
  flat step for the search to exploit.

Everything beyond stays as the map had it, and unknown stays unknown: unknown
is a statement about what the lidar has seen, not about what is there, and
flattening it into "occupied" would make the robot unable to plan into any
room it has not already entered.
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String


class GlobalCostmapNode(Node):
    def __init__(self):
        super().__init__('global_costmap_node')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('costmap_topic', '/global_costmap')
        self.declare_parameter('robot_radius_m', 0.205)
        self.declare_parameter('inflation_radius_m', 0.45)
        self.declare_parameter('cost_scaling', 3.0)
        self.declare_parameter('lethal_threshold', 65)
        self.declare_parameter('update_period_s', 1.0)
        self.declare_parameter('inflate_unknown_edges', True)

        g = self.get_parameter
        self._rr = float(g('robot_radius_m').value)
        self._ir = max(float(g('inflation_radius_m').value), self._rr + 1e-3)
        self._scale = float(g('cost_scaling').value)
        self._lethal = int(g('lethal_threshold').value)
        self._edge = bool(g('inflate_unknown_edges').value)

        self._pending = None
        self._last = 0.0

        latched = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._pub = self.create_publisher(
            OccupancyGrid, str(g('costmap_topic').value), latched)
        self._status = self.create_publisher(String, '/global_costmap/status',
                                             10)
        self.create_subscription(OccupancyGrid, str(g('map_topic').value),
                                 self._map_cb, latched)
        self.create_timer(float(g('update_period_s').value), self._tick)

        self.get_logger().info(
            'global_costmap_node up — lethal within %.2f m, inflated to '
            '%.2f m' % (self._rr, self._ir))

    def _map_cb(self, msg):
        # Store and process on the timer rather than inline: inflation on a
        # 600x600 grid takes tens of milliseconds, and doing it in the
        # subscription callback blocks the executor for every map update.
        self._pending = msg

    def _tick(self):
        msg = self._pending
        if msg is None:
            return
        self._pending = None
        t0 = time.perf_counter()

        w, h = msg.info.width, msg.info.height
        res = msg.info.resolution
        src = np.asarray(msg.data, dtype=np.int16).reshape(h, w)

        occupied = src >= self._lethal
        if self._edge:
            # An unknown cell touching an occupied one is almost always the
            # shadow behind that obstacle — the lidar could not see past it.
            # Treating those as obstacles for inflation purposes stops the
            # planner threading a route through the blind side of a wall.
            unknown = src < 0
            pad = np.zeros_like(occupied)
            pad[1:, :] |= occupied[:-1, :]
            pad[:-1, :] |= occupied[1:, :]
            pad[:, 1:] |= occupied[:, :-1]
            pad[:, :-1] |= occupied[:, 1:]
            occupied = occupied | (unknown & pad)

        dist = _edt(occupied) * res

        out = src.copy()
        out[dist <= self._rr] = 100
        band = (dist > self._rr) & (dist <= self._ir)
        if band.any():
            span = max(self._ir - self._rr, 1e-6)
            frac = (dist[band] - self._rr) / span
            val = (self._lethal - 1) * np.exp(-self._scale * frac)
            out[band] = np.maximum(out[band], val.astype(np.int16))
        # Unknown stays unknown.
        out[src < 0] = -1

        cm = OccupancyGrid()
        cm.header.stamp = self.get_clock().now().to_msg()
        cm.header.frame_id = msg.header.frame_id
        cm.info = msg.info
        cm.data = out.ravel().tolist()
        self._pub.publish(cm)

        dt = (time.perf_counter() - t0) * 1000.0
        s = String()
        s.data = ('%dx%d @ %.3f m, %d lethal cells, %.0f ms'
                  % (w, h, res, int((out >= 100).sum()), dt))
        self._status.publish(s)
        self.get_logger().debug(s.data)


def _edt(mask):
    """Exact Euclidean distance transform, in cells, from True cells.

    Felzenszwalb and Huttenlocher's separable algorithm: a 1-D transform down
    the columns, then along the rows, each O(n) via a lower-envelope scan.
    The whole thing is O(cells) and vectorises, which matters because the
    naive alternative — stamping a disc at every obstacle — is O(obstacles x
    disc area) and visibly stalls the node on a cluttered map.
    """
    INF = 1e12
    f = np.where(mask, 0.0, INF)
    d = np.empty_like(f)
    for c in range(f.shape[1]):
        d[:, c] = _edt1d(f[:, c])
    out = np.empty_like(d)
    for r in range(d.shape[0]):
        out[r, :] = _edt1d(d[r, :])
    return np.sqrt(out)


def _edt1d(f):
    n = len(f)
    d = np.empty(n)
    v = np.zeros(n, dtype=np.int64)
    z = np.empty(n + 1)
    k = 0
    v[0] = 0
    z[0] = -1e20
    z[1] = 1e20
    for q in range(1, n):
        while True:
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q
                                                             - 2.0 * v[k])
            if s <= z[k]:
                k -= 1
                if k < 0:
                    k = 0
                    break
            else:
                break
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = 1e20
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
    return d


def main(args=None):
    rclpy.init(args=args)
    node = GlobalCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
