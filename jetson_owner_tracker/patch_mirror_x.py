"""Apply on the Jetson: fixes steering direction (mirror_x + live angular_sign).

Run from ~/owner_tracker/owner_tracker:
    python3 patch_mirror_x.py
Then rebuild the ROS package:
    cd ~/ros2_ws && colcon build --packages-select followme_ros --symlink-install && source install/setup.bash
"""

def edit(path, old, new):
    s = open(path).read()
    if new in s:
        print("already done:", path); return
    assert s.count(old) == 1, f"pattern not found in {path}: {old[:70]!r}"
    open(path, "w").write(s.replace(old, new)); print("patched:", path)


# ---------------------------------------------------------------- config.py
edit("followme/config.py",
"    bbox_range_fallback: bool = False       # RGB-only testing: range from box height (not in paper)\n",
"    mirror_x: bool = False                  # flip left/right at the source if the camera reads mirrored\n"
"    bbox_range_fallback: bool = False       # RGB-only testing: range from box height (not in paper)\n")

# ------------------------------------------------------------ localization.py
edit("followme/localization.py",
"""def target_position(depth_m: np.ndarray, mask: np.ndarray, intr: CameraIntrinsics,
                    cfg: LocalizationConfig) -> Optional[np.ndarray]:
    \"\"\"``c_p``: 3-D centroid of the segmented cloud, or None if too few points.\"\"\"
    cloud = segmented_point_cloud(depth_m, mask, intr, cfg)
    if len(cloud) < cfg.min_points:
        return None
    return np.median(cloud, axis=0) if cfg.centroid == "median" else cloud.mean(axis=0)""",
"""def target_position(depth_m: np.ndarray, mask: np.ndarray, intr: CameraIntrinsics,
                    cfg: LocalizationConfig) -> Optional[np.ndarray]:
    \"\"\"``c_p``: 3-D centroid of the segmented cloud, or None if too few points.\"\"\"
    cloud = segmented_point_cloud(depth_m, mask, intr, cfg)
    if len(cloud) < cfg.min_points:
        return None
    p = np.median(cloud, axis=0) if cfg.centroid == "median" else cloud.mean(axis=0)
    if cfg.mirror_x:
        p = p.copy()
        p[0] = -p[0]
    return p""")

edit("followme/localization.py",
"""def position_from_bbox(bbox: np.ndarray, intr: CameraIntrinsics, person_height_m: float = 1.7) -> np.ndarray:
    \"\"\"RGB-only fallback (NOT in the paper): pinhole range from the box height.

    ``z = fy * H / h_px``; only for testing the pipeline on a webcam without
    depth. Partial occlusion or a crouching person corrupts it.
    \"\"\"
    x1, y1, x2, y2 = [float(v) for v in bbox]
    z = intr.fy * person_height_m / max(1.0, y2 - y1)
    u, v = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    return intr.deproject(np.array([u]), np.array([v]), np.array([z]))[0]""",
"""def position_from_bbox(bbox: np.ndarray, intr: CameraIntrinsics, person_height_m: float = 1.7,
                       mirror_x: bool = False) -> np.ndarray:
    \"\"\"RGB-only fallback (NOT in the paper): pinhole range from the box height.

    ``z = fy * H / h_px``; only for testing the pipeline on a webcam without
    depth. Partial occlusion or a crouching person corrupts it.
    \"\"\"
    x1, y1, x2, y2 = [float(v) for v in bbox]
    z = intr.fy * person_height_m / max(1.0, y2 - y1)
    u, v = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    p = intr.deproject(np.array([u]), np.array([v]), np.array([z]))[0]
    if mirror_x:
        p = p.copy()
        p[0] = -p[0]
    return p""")

# ---------------------------------------------------------------- pipeline.py
edit("followme/pipeline.py",
"                    p_cam = position_from_bbox(persons[cand].bbox, self.intrinsics, loc.assumed_person_height_m)\n",
"                    p_cam = position_from_bbox(persons[cand].bbox, self.intrinsics, loc.assumed_person_height_m,\n"
"                                               loc.mirror_x)\n")
edit("followme/pipeline.py",
"                p = position_from_bbox(persons[i].bbox, self.intrinsics, loc.assumed_person_height_m)\n",
"                p = position_from_bbox(persons[i].bbox, self.intrinsics, loc.assumed_person_height_m,\n"
"                                       loc.mirror_x)\n")

# ------------------------------------------------------------ followme_node.py
p = "ros2/followme_ros/followme_ros/followme_node.py"
edit(p, "from rclpy.node import Node\n", "from rclpy.node import Node\nfrom rcl_interfaces.msg import SetParametersResult\n")

edit(p,
'        p("angular_sign", 1.0)                     # -1.0 flips steering (base turns the other way)\n',
'        p("angular_sign", 1.0)                     # -1.0 flips steering (base turns the other way)\n'
'        p("mirror_x", False)                       # flip left/right at the vision source (see below)\n')

edit(p,
"""        self.get_logger().info(f"FollowMe ready (device {self.device}, safety circle "
                               f"{self.controller.safety_radius:.2f} m). Call ~/calibrate with the target alone.")""",
"""        self.cfg.localization.mirror_x = bool(g("mirror_x"))
        self.add_on_set_parameters_callback(self._on_param_change)
        self.get_logger().info(f"FollowMe ready (device {self.device}, safety circle "
                               f"{self.controller.safety_radius:.2f} m). Call ~/calibrate with the target alone.")

    def _on_param_change(self, params):
        \"\"\"Make angular_sign and mirror_x ACTUALLY live-settable (previously angular_sign was only
        read once at startup: 'ros2 param set' would report success but change nothing running).\"\"\"
        for p in params:
            if p.name == "mirror_x":
                self.cfg.localization.mirror_x = bool(p.value)
                self.get_logger().info(f"mirror_x -> {p.value} (live)")
            elif p.name == "angular_sign":
                if hasattr(self.backend, "angular_sign"):
                    self.backend.angular_sign = float(p.value)
                    self.get_logger().info(f"angular_sign -> {p.value} (live)")
                else:
                    self.get_logger().warn("angular_sign has no effect: control_mode is not cmd_vel")
        return SetParametersResult(successful=True)""")

print("\nAll patches applied. Now rebuild:")
print("  cd ~/ros2_ws && colcon build --packages-select followme_ros --symlink-install && source install/setup.bash")
