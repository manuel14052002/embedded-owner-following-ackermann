#!/usr/bin/env python3
"""A2RBOT steering calibration - two short tests that remove all left/right guessing.

Test 1 - WHEELS (FollowMe must be STOPPED, the serial bridge must be RUNNING):
    python3 steer_test.py wheels
  Sends "turn LEFT" (the ROS standard: angular.z > 0) with zero speed for 3 s.
  You watch the front wheels STANDING BEHIND THE ROBOT, looking where the camera looks,
  so "left" means the same for you and for the robot. Answer y/n -> ANGULAR_SIGN.

Test 2 - CAMERA (FollowMe must be RUNNING and you must be its owner):
    python3 steer_test.py vision
  Prints live where FollowMe thinks you are, from the robot's point of view. You walk to
  the robot's left-wheel side. If it says RIGHT, the image is mirrored -> MIRROR_X=true.

At the end it prints the exact lines to put into ~/start_a2rbot.sh.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Optional

LOCK = "/tmp/followme_node.lock"


def recommended(wheels_turned_left: Optional[bool], camera_said_left: Optional[bool]) -> dict:
    """Settings from the two observations (None = test not done)."""
    out = {}
    if wheels_turned_left is not None:
        out["ANGULAR_SIGN"] = "1.0" if wheels_turned_left else "-1.0"
    if camera_said_left is not None:
        out["MIRROR_X"] = "false" if camera_said_left else "true"
    return out


def followme_running() -> Optional[int]:
    try:
        pid = int(Path(LOCK).read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        import os
        os.kill(pid, 0)
        return pid
    except OSError:
        return None


def ask(question: str) -> bool:
    while True:
        a = input(f"{question} [y/n]: ").strip().lower()
        if a in ("y", "yes"):
            return True
        if a in ("n", "no"):
            return False


def wheels_test() -> Optional[bool]:  # pragma: no cover - needs ROS + hardware
    pid = followme_running()
    if pid:
        print(f"!! FollowMe is running (pid {pid}) and would fight over /cmd_vel. Stop it first:\n"
              f"     ~/stop_a2rbot.sh\n   (keep the serial bridge running on the other PC)")
        return None
    import rclpy
    from geometry_msgs.msg import Twist

    rclpy.init()
    node = rclpy.create_node("steer_test")
    pub = node.create_publisher(Twist, "/cmd_vel", 10)
    print("\nWait for a subscriber on /cmd_vel (the serial bridge)...", end="", flush=True)
    t0 = time.time()
    while pub.get_subscription_count() == 0 and time.time() - t0 < 10:
        rclpy.spin_once(node, timeout_sec=0.2)
        print(".", end="", flush=True)
    print()
    if pub.get_subscription_count() == 0:
        print("!! nobody subscribes to /cmd_vel - is the serial bridge running (same ROS_DOMAIN_ID)?")
        node.destroy_node()
        rclpy.shutdown()
        return None
    print("\n>>> STAND BEHIND THE ROBOT, looking in the same direction as the camera.")
    input(">>> Press Enter, then watch the FRONT WHEELS for 3 seconds...")
    tw = Twist()
    tw.angular.z = 0.5                         # ROS standard: positive = turn LEFT
    end = time.time() + 3.0
    while time.time() < end:
        pub.publish(tw)
        time.sleep(0.05)
    stop = Twist()
    for _ in range(10):
        pub.publish(stop)
        time.sleep(0.05)
    left = ask("Did the front wheels point to YOUR LEFT (= the robot's left)?")
    node.destroy_node()
    rclpy.shutdown()
    return left


def vision_test() -> Optional[bool]:  # pragma: no cover - needs ROS + hardware
    if not followme_running():
        print("!! FollowMe is not running. Start it (~/start_a2rbot.sh) and claim the robot first.")
        return None
    import rclpy
    from std_msgs.msg import Float32

    rclpy.init()
    node = rclpy.create_node("steer_test_vision")
    last = {"theta": None, "t": 0.0}

    def cb(msg):
        last["theta"], last["t"] = float(msg.data), time.time()
    node.create_subscription(Float32, "/owner_bearing", cb, 10)
    print("\n>>> You must be the OWNER (claimed). Stand ~2 m in front of the camera.")
    print(">>> Then walk slowly toward the side of the robot's LEFT WHEEL")
    print("    (the robot's left = your RIGHT while you face the robot).")
    print(">>> Watch the line below; press Ctrl+C when you are clearly on that side.\n")
    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.1)
            th = last["theta"]
            if th is None or time.time() - last["t"] > 1.0:
                txt = "owner not seen (claim the robot / step into view)"
            else:
                deg = math.degrees(th)
                side = "STRAIGHT" if abs(deg) < 5 else ("LEFT" if deg > 0 else "RIGHT")
                txt = f"FollowMe sees you on the ROBOT'S {side:8s} ({deg:+5.0f} deg)"
            print("\r" + txt.ljust(70), end="", flush=True)
    except KeyboardInterrupt:
        print()
    node.destroy_node()
    rclpy.shutdown()
    return ask("When you stood at the robot's LEFT wheel, did it say ROBOT'S LEFT?")


def main() -> int:  # pragma: no cover - interactive
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("wheels", "vision"):
        print(__doc__)
        return 1
    res = wheels_test() if mode == "wheels" else vision_test()
    if res is None:
        return 2
    settings = recommended(res if mode == "wheels" else None, res if mode == "vision" else None)
    print("\n=== RESULT ===")
    for k, v in settings.items():
        print(f"  {k}={v}")
        print(f"  apply:  sed -i 's/^{k}=[^ ]*/{k}={v}/' ~/start_a2rbot.sh")
    print("  then:   ~/stop_a2rbot.sh && ~/start_a2rbot.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
