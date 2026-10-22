#!/usr/bin/env python3
"""
map_saver_node.py
=================
Writes /map to disk as a PGM image plus a YAML sidecar — the standard ROS map
format, readable by map_server, by GIMP, and by anything else that has ever
opened a Netpbm file.

Written by hand rather than depending on nav2_map_server.  nav2 is a large
dependency to pull onto a Raspberry Pi for the sake of one file writer, and
the format is about forty lines of code:

    P5              binary greyscale
    width height
    255             max value
    <width*height bytes, row-major, TOP row first>

The one thing that catches people is the row order.  An OccupancyGrid's first
row is the one at the *lowest* y — the origin is the bottom-left corner. A PGM
image's first row is the *top*. So the array has to be flipped vertically on
the way out, and flipped back on the way in. Get this wrong and your map is
mirrored, which is surprisingly hard to notice on a symmetric building and
then impossible to debug.

Pixel convention, matching map_server:

    255 (white)  free
    000 (black)  occupied
    205 (grey)   unknown

Trigger a save by publishing the destination path:

    ros2 topic pub --once /map_save std_msgs/String "{data: /home/pi/maps/lab}"

or leave `auto_save_period_s` non-zero to checkpoint while mapping, which is
worth doing — a two-hour mapping run ended by a flat battery is a sad thing.
"""

import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String, Bool


class MapSaverNode(Node):
    def __init__(self):
        super().__init__('map_saver_node')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('default_path', '~/maps/a2r_map')
        self.declare_parameter('occupied_threshold', 0.65)
        self.declare_parameter('free_threshold', 0.25)
        self.declare_parameter('auto_save_period_s', 0.0)
        self.declare_parameter('negate', False)

        g = self.get_parameter
        self._default = os.path.expanduser(str(g('default_path').value))
        self._occ = float(g('occupied_threshold').value)
        self._free = float(g('free_threshold').value)
        self._negate = bool(g('negate').value)

        self._map = None
        self._saves = 0

        latched = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._result = self.create_publisher(String, '/map_save/result', 10)

        self.create_subscription(OccupancyGrid, str(g('map_topic').value),
                                 self._map_cb, latched)
        self.create_subscription(String, '/map_save', self._save_cb, 10)
        self.create_subscription(Bool, '/map_save/trigger',
                                 self._trigger_cb, 10)

        period = float(g('auto_save_period_s').value)
        if period > 0.0:
            self.create_timer(period, lambda: self._save(self._default))
            self.get_logger().info('auto-saving to %s every %.0f s'
                                   % (self._default, period))

        self.get_logger().info(
            'map_saver_node up — publish a path on /map_save, or true on '
            '/map_save/trigger to write %s' % self._default)

    def _map_cb(self, msg):
        self._map = msg

    def _save_cb(self, msg: String):
        self._save(os.path.expanduser(msg.data.strip()) or self._default)

    def _trigger_cb(self, msg: Bool):
        if msg.data:
            self._save(self._default)

    # ------------------------------------------------------------------
    def _save(self, stem):
        if self._map is None:
            self._report('no map received yet — nothing to save', ok=False)
            return
        if not stem:
            stem = self._default
        if stem.endswith('.yaml') or stem.endswith('.pgm'):
            stem = os.path.splitext(stem)[0]

        try:
            d = os.path.dirname(stem)
            if d:
                os.makedirs(d, exist_ok=True)

            msg = self._map
            w, h = msg.info.width, msg.info.height
            res = msg.info.resolution
            data = np.asarray(msg.data, dtype=np.int16).reshape(h, w)

            img = np.full((h, w), 205, dtype=np.uint8)      # unknown
            known = data >= 0
            prob = data.astype(np.float32) / 100.0
            img[known & (prob <= self._free)] = 254         # free
            img[known & (prob >= self._occ)] = 0            # occupied
            # Anything known but in between stays grey: map_server would
            # treat it as unknown too, and pretending to a certainty the
            # mapper never had is how you get a robot confidently driving at
            # a half-observed wall.

            # OccupancyGrid row 0 is the lowest y; PGM row 0 is the top.
            img = np.flipud(img)

            pgm = stem + '.pgm'
            with open(pgm, 'wb') as fh:
                fh.write(b'P5\n')
                fh.write(b'# Created by a2r_bot_slam map_saver_node\n')
                fh.write(('%d %d\n255\n' % (w, h)).encode('ascii'))
                fh.write(img.tobytes())

            yaml_path = stem + '.yaml'
            with open(yaml_path, 'w') as fh:
                fh.write('image: %s\n' % os.path.basename(pgm))
                fh.write('resolution: %.6f\n' % res)
                fh.write('origin: [%.6f, %.6f, %.6f]\n'
                         % (msg.info.origin.position.x,
                            msg.info.origin.position.y, 0.0))
                fh.write('negate: %d\n' % (1 if self._negate else 0))
                fh.write('occupied_thresh: %.3f\n' % self._occ)
                fh.write('free_thresh: %.3f\n' % self._free)
                fh.write('mode: trinary\n')

            self._saves += 1
            self._report('saved %s (%dx%d @ %.3f m/px)'
                         % (stem, w, h, res), ok=True)
        except Exception as exc:                            # noqa: BLE001
            self._report('save failed: %s: %s' % (type(exc).__name__, exc),
                         ok=False)

    def _report(self, text, ok):
        m = String()
        m.data = text
        self._result.publish(m)
        if ok:                                   # a2r_fix_freeze: one call site per severity
            self.get_logger().info(text)
        else:
            self.get_logger().error(text)


def main(args=None):
    rclpy.init(args=args)
    node = MapSaverNode()
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
