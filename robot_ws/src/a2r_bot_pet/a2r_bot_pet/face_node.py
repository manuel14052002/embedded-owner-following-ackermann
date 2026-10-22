#!/usr/bin/env python3
"""
face_node.py
============
Runs the animated face and listens for which expression to show.

-----------------------------------------------------------------------------
The threading arrangement, which is the only interesting thing here
-----------------------------------------------------------------------------
pygame wants to own the main thread: on most platforms the event loop and the
display surface must live there, and calling `pygame.display.flip()` from a
worker thread produces anything from a black screen to a segfault depending on
the SDL backend.

rclpy also wants to own a thread to spin its executor.

They cannot both have the main thread, so:

    main thread     pygame — events, animation, rendering, vsync
    worker thread   rclpy executor — subscriptions

The only shared state is a single string (the requested expression) plus a
couple of flags, each written by one thread and read by the other. Python
guarantees that assigning a name is atomic, so no lock is needed for a plain
string swap — and adding one would risk the render loop blocking on a
subscription callback, which at 90 fps you would see as stutter.

The render loop never blocks on ROS, and ROS never touches the display.

-----------------------------------------------------------------------------
Headless
-----------------------------------------------------------------------------
If no display is available the node does not die; it logs the requested
expressions instead. That matters because this package gets launched as part
of follow mode over SSH, and a face node that takes the whole launch file down
because $DISPLAY is unset is a bad trade for a robot that was otherwise
working.
"""

import os
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from std_msgs.msg import String, Bool


class FaceNode(Node):
    """ROS half: subscribes, and hands the render loop a string."""

    def __init__(self):
        super().__init__('face_node')

        self.declare_parameter('face_topic', '/pet/face')
        self.declare_parameter('start_expression', 'idle')
        self.declare_parameter('fullscreen', True)
        self.declare_parameter('screen_width', 800)
        self.declare_parameter('screen_height', 480)
        self.declare_parameter('fps', 60)
        self.declare_parameter('show_mouth', True)
        self.declare_parameter('auto_sequence', False)
        self.declare_parameter('sdl_videodriver', '')
        self.declare_parameter('sdl_fbdev', '')

        g = self.get_parameter
        self.requested = str(g('start_expression').value)
        self.quit_requested = False
        self.shown = ''

        latched = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._current_pub = self.create_publisher(
            String, '/pet/current_expression', latched)

        self.create_subscription(String, str(g('face_topic').value),
                                 self._face_cb, 10)
        self.create_subscription(Bool, '/pet/quit', self._quit_cb, 10)

    def _face_cb(self, msg: String):
        name = msg.data.strip()
        if name:
            self.requested = name

    def _quit_cb(self, msg: Bool):
        if msg.data:
            self.quit_requested = True

    def announce(self, name):
        if name == self.shown:
            return
        self.shown = name
        m = String()
        m.data = name
        self._current_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = FaceNode()
    g = node.get_parameter

    # SDL configuration has to happen before pygame initialises its display.
    # On a Pi driving a bare framebuffer LCD with no desktop, these two are
    # the difference between a working screen and "No available video device".
    drv = str(g('sdl_videodriver').value)
    fb = str(g('sdl_fbdev').value)
    if drv:
        os.environ['SDL_VIDEODRIVER'] = drv
    if fb:
        os.environ['SDL_FBDEV'] = fb

    # Spin ROS on a worker so pygame keeps the main thread.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    renderer = None
    try:
        from a2r_bot_pet import robot_face
    except Exception as exc:                                # noqa: BLE001
        node.get_logger().error('cannot import robot_face: %s' % exc)
        robot_face = None

    if robot_face is not None:
        try:
            renderer = robot_face.FaceRenderer(
                width=int(g('screen_width').value),
                height=int(g('screen_height').value),
                fullscreen=bool(g('fullscreen').value),
                start_expression=str(g('start_expression').value),
                auto_sequence=bool(g('auto_sequence').value),
                show_mouth=bool(g('show_mouth').value),
                caption='A2R_bot')
            node.get_logger().info(
                'face up — %s, expressions: %s'
                % ('fullscreen' if bool(g('fullscreen').value) else 'windowed',
                   ', '.join(robot_face.list_expressions())))
        except Exception as exc:                            # noqa: BLE001
            node.get_logger().warn(
                'no display (%s) — running headless, expressions will be '
                'logged instead' % exc)
            renderer = None

    fps = int(g('fps').value)
    try:
        if renderer is None:
            _headless(node)
        else:
            _render_loop(node, renderer, fps)
    except KeyboardInterrupt:
        pass
    finally:
        if renderer is not None:
            renderer.close()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _render_loop(node, renderer, fps):
    """Main thread: pygame owns this and never blocks on ROS."""
    unknown_warned = set()
    while rclpy.ok() and not node.quit_requested:
        dt = renderer.tick(fps)
        if not renderer.pump_events():
            break

        want = node.requested
        if want != renderer.expression_name:
            if not renderer.set_expression(want):
                if want not in unknown_warned:
                    unknown_warned.add(want)
                    node.get_logger().warn(
                        'unknown expression "%s" — ignoring. Known: %s'
                        % (want, ', '.join(
                            __import__('a2r_bot_pet.robot_face',
                                       fromlist=['x']).list_expressions())))
        node.announce(renderer.expression_name)
        renderer.step(dt)


def _headless(node):
    """No display: keep the node alive and report what it would show."""
    last = None
    while rclpy.ok() and not node.quit_requested:
        if node.requested != last:
            last = node.requested
            node.announce(last)
            node.get_logger().info('[headless] expression -> %s' % last)
        time.sleep(0.1)


if __name__ == '__main__':
    main()
