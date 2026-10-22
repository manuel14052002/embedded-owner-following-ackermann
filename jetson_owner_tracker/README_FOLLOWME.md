# FollowMe — implementation of arXiv:2311.12992

Rollo, Zunino, Raiola, Amadio, Ajoudani, Tsagarakis, *FollowMe: a Robust Person
Following Framework Based on Visual Re-Identification and Gestures* (2023),
implemented as the `followme/` package next to the existing `owner_tracking/`
tracker, plus a ROS 2 Humble / Nav2 wrapper in `ros2/followme_ros`.

## 0. Deployment quick start (v8)

**Start and check:**
```bash
~/start_a2rbot.sh            # camera + FollowMe + face + voice; Ctrl+C stops everything
~/a2rbot_check.sh            # in a 2nd terminal: PASS/WARN/FAIL for every link of the chain
```

**Calibrate the steering direction once.** Each test gives one definite answer, and no
left/right guessing is needed.
```bash
~/stop_a2rbot.sh && python3 ~/steer_test.py wheels     # bridge running, FollowMe stopped
~/start_a2rbot.sh  (claim)  && python3 ~/steer_test.py vision
```

**Configuration.** Everything is in `config/followme_deploy.yaml`, a complete file.

**Safety layers**, each of which removes forward motion on its own:
1. The follow distance (0.6 m) and the safety circle (0.5 m).
2. A camera-depth hard stop (0.5 m straight ahead).
3. A lidar front-sector stop (0.5 m within ±45°).
4. Driving only on a fresh real detection (0.5 s).
5. A 0.5 s cmd_vel watchdog, plus the bridge's own 0.3 s watchdog.
6. The voice "circular" arc obeys layers 2 and 3.

**Other deployment behaviour:**
* A single `followme_node` is enforced by a PID lock.
* The claimed owner is saved automatically and reloaded after a restart.

## 1. Paper → code

| Paper | What | Code |
|---|---|---|
| III-A, Fig. 1b | Instance segmentation (COCO *person*), image × mask to remove the background | `followme/segmentation.py` |
| III-B, VI | MMT pre-trained IBN-ResNet-50, MSMT17, D = 256 | `followme/reid.py`, `scripts/followme_train_mmt_pretrain.py`, `scripts/followme_export_mmt_onnx.py` |
| III-B Calibration | 2/3 → μ, σ; 1/3 → threshold set | `followme/calibration.py` (`fit_target_model`, `CalibrationCollector`) |
| Eq. 1 | d = √(1/D Σ ((x*ᵢ − μᵢ)/σᵢ)²) | `TargetModel.distances` |
| III-B | λ_d = μ_d + 2σ_d; target = argmin d if d ≤ λ_d | `fit_target_model`, `identify` |
| III-C, Fig. 1d | Masked depth → point cloud → centroid c_p | `followme/localization.py` (`target_position`) |
| III-C | Kalman filter, stops after t_exp | `TargetKalmanFilter` |
| Eq. 2 | m_p = m_T_c · c_p, goal on the map XY plane, heading θ | `transform_point`, `compute_goal`, `heading_angle` |
| III-D, Fig. 4 | MediaPipe 21 key points → 63-d vector → RBF one-vs-one SVM, classes Wait/Follow/Others | `followme/gestures.py` |
| III-D | ξ = 5 equal consecutive outputs before a command | `ConsecutiveFilter` |
| IV, Eq. 3 | Nav stack goals, safety circle d_safe = 1.4·v_max·t_exp | `followme/navigation.py`, `ros2/followme_ros` |
| V, Fig. 5 | Steady / Follow / Search / Wait with α, β, γ | `followme/state_machine.py` |
| VI-A, Fig. 6a, Table I | Gesture evaluation | `scripts/followme_collect_gestures.py`, `scripts/followme_train_gestures.py` |
| VI-B, Fig. 2, Fig. 6b, Table I | Re-ID protocol | `scripts/followme_eval_reid.py`, `followme/evaluation.py` |
| VI-C | Following run, timing (7–10 Hz) | `followme_app.py` CSV log, ROS node `~/state`, `~/target_pose` |

All paper constants are defaults in `followme/config.py` / `config/followme.yaml`
(D = 256, 2/3 split, k = 2, ξ = 5, t_exp = 3 s, v_max = 0.3 m/s → safety radius 1.26 m;
the paper rounds to 1.25 m).

## 2. State machine as implemented

α = a filtered target position exists (the target was re-identified within t_exp),
β = the last target position is inside the safety circle, γ = gesture command.

```
STEADY --α-->           FOLLOW        FOLLOW --β & !α-->  STEADY
FOLLOW --!α & !β-->     SEARCH        SEARCH --α-->       FOLLOW
SEARCH --full turn, !α--> STEADY
FOLLOW --γ wait-->      WAIT          WAIT   --γ follow--> FOLLOW
FOLLOW with α & β: stay in FOLLOW, delete the goal and stop (Sect. IV)
```

It starts in `WAIT`, as in the experiments where the robot waits for a Follow
gesture. Search turns toward the side where the target was last seen.

## 3. Deviations from the paper (and why)

* **YOLO11-seg instead of YOLACT++.** YOLACT++ needs the DCNv2 CUDA extension,
  which does not build on current PyTorch or JetPack. The replacement has the
  same interface (COCO person masks) and is faster. Any Ultralytics `-seg`
  model works, including TensorRT `.engine` files.
* **MMT weights are not shipped.** Train them with
  `scripts/followme_train_mmt_pretrain.py` on MSMT17, or load any MMT
  `ResNetIBN('50a')` checkpoint. `backend: torchreid` with OSNet-MSMT17 is a
  ready-made alternative (not the paper's network).
* **σ floor.** σᵢ is clamped to ≥ 5 % of mean(σ) so that a near-constant
  feature dimension cannot dominate Eq. 1. This has no effect on dense 256-d
  features.
* **Calibration hygiene** (paper Sect. VII names calibration quality as the
  main weakness):
  * frames showing more than one person are rejected;
  * samples are spaced in time;
  * distance coverage is measured from depth, with a warning if you stayed
    at one distance.
* **Gestures only from the target's crop** (Fig. 1 draws Re-ID → gesture detection).
  The 63-d vector is made independent of hand position and size, and left
  hands are mirrored so one classifier covers both hands.
* **Mask erosion (5 px).** Keeps background depth at the mask edge out of the centroid.
  `centroid: median` and `depth_inlier_band_m` are extra robust options; the
  default `mean` is the paper's choice.
* **Wait from any state.** The Wait gesture also stops the robot from STEADY and SEARCH.
  Fig. 5 draws it only from FOLLOW. Set `wait_from_any_state: false` for the strict figure.
* **Ackermann bases.** A car-like base cannot rotate in place, so set
  `navigation.search_linear_speed > 0` and Search drives an arc instead
  (`config/followme_a2rbot.yaml`).
* **RGB-only fallback** (`--rgb-only`): range is estimated from box height, for
  desk testing only. It is not in the paper and not for driving.

## 3b. Identity extensions (not in the paper, switched on in `config/followme_osnet.yaml`)

Tested live with OSNet on the Jetson: the paper's Eq. 1 could not separate a
white-shirt target from a green-shirt stranger (the stranger's distance was
below the target's own mean). The extensions below are off by default, so
`config/followme.yaml` still reproduces the paper.

* **Re-ID and calibration settings:**
  * **`reid.backend: torchreid`** uses the fast NumPy preprocessing path
    (about 44 ms instead of 124 ms for 3 people on an Orin Nano). cuDNN
    auto-tuning stays off, because it runs the GPU out of memory.
  * **`calibration.duration_s`** spreads the samples over a fixed time.
    The **E** key finishes calibration early.
* **`calibration.metric`** chooses how a candidate is compared with the target:
  * **`eq1`**: the paper's method.
  * **`cosine`**: cosine distance to the target's average fingerprint.
  * **`both`**: must pass Eq. 1 and cosine.
  * **`gallery`**: the recommended option.
    * *Multi-view gallery:* the distance is 1 − mean of the top-k cosine
      similarities to the stored calibration views, so front and back views
      each have their own references.
    * *Known strangers:* people seen next to a confidently identified target
      go into a stranger bank. A candidate must look more like the target
      than like those strangers, by `identity.negative_margin`.
    * *Online views:* confident, unoccluded, non-duplicate target views are
      added while following (at most one per second).
  * All thresholds are learned from the threshold set as mean + k·std.
* **`localization.confirm_frames`**: a new target must pass N consecutive
  frames at a consistent position, which removes one-frame "blinks".
* **`localization.gate_m`**: while tracked, a match is accepted only near the
  Kalman prediction. The gate grows by `max_person_speed_mps` for every second
  the target is unseen.

The overlay shows the active threshold and each person's distance. In gallery
mode it also shows how many views were added, how many strangers are known,
and how many candidates the stranger bank has blocked.

## 3c. Ownership by gesture (not in the paper; `ownership.enabled: true` in `config/followme_osnet.yaml`)

```
NO_OWNER --one person holds a RAISED open palm 2 s--> CLAIMING (20 s auto-calibration of that person;
   ^                                                    everyone else seen -> known strangers)
   |                                                         | done                | claimant lost 2 s
   |                                                         v                     v
   +--- owner holds a RAISED open palm 3 s, or unseen 60 s --- OWNED (FollowMe)   NO_OWNER
```

The ownership states:
* **NO_OWNER.** The robot is stopped, and the hands of the 4 largest people
  are checked. If two people raise a hand at the same time, both are ignored.
* **CLAIMING.** Robot stays stopped while the claimant's views are collected
  (automatic calibration of that person).
* **OWNED.** Normal FollowMe:
  * following starts automatically once the new owner is identified;
  * a fist means FOLLOW and a low open palm means WAIT;
  * other people's raised hands are ignored;
  * a raised palm first acts as WAIT (the robot stops), then releases the
    robot after 3 s.

Other rules:
* **"Raised"** means the wrist is in the top 25 % of the person's box and
  horizontally inside it. The open palm is the gesture classifier's WAIT
  class, so no extra training is needed.
* **After a release** there is a 3 s cooldown, so the same raised hand does
  not immediately re-claim the robot.
* **Manual control still works:** R (or the `~/reset` service) releases the
  robot, and C still calibrates manually.
* **Requirement:** the gesture classifier (`models/gesture_svm.joblib`).

## 3d. Direct cmd_vel control and voice integration (ROS node)

```
 RealSense (aligned depth) --> followme_node --/cmd_vel (Twist)--> serial_bridge --> ESP32 (Ackermann)
 voice system --/voice_command (JSON {"action": ...})--> followme_node
              --/robot_state_cmd ("FOLLOW"/"STOP")-->
 followme_node --/owner_distance (Float32, -1 = unseen)--> voice system
               --/robot_state (String: WAIT/FOLLOW/SEARCH/STEADY)-->
```

* **Control mode.** `control_mode:=cmd_vel` (the default) drives the robot
  directly from the owner's position in `base_link` and needs no map or Nav2.
  * Speed: v = k_linear·(distance − follow_distance), slowed by cos(bearing)
    and ramped with acceleration limits.
  * Turning: w = k_angular·bearing.
  * Follow distance: 0.3 m beyond the safety circle, unless
    `navigation.follow_distance_m` sets it.
  * A watchdog sends zero velocity if perception stalls for 0.5 s.
  * **This mode has no obstacle avoidance.** Use `control_mode:=nav2`
    where obstacles matter.
* **TF.** The node uses TF `base_link` ← camera optical frame. Without that
  transform it assumes a forward-facing camera at the base origin.
* **Voice commands** are accepted as JSON `{"action"|"command"|"cmd": ...}`
  or as plain text:
  * `follow`, `start`, `come` → FOLLOW;
  * `stop`, `wait`, `stay` → WAIT;
  * `release`, `release_owner` → release ownership;
  * `enroll`, `enroll_owner`, `calibrate` → release, then calibrate whoever
    is alone in view.
  * Other actions are ignored.
* **Only one node may publish `/cmd_vel` while following.** Do not run the
  old `tracker_node`. FollowMe publishes zero velocity only when it stops, so
  voice or teleop commands still work while it is idle.

## 3e. voice_bridge.py integration (phone page on port 8090)

**Commands.** `voice_bridge.py` publishes `{"action": ...}` on `/voice_command`:

| Bridge action | FollowMe behaviour |
|---|---|
| `wake_trigger`, or `follow` while nobody owns the robot | Arms a claim for `ownership.voice_arm_s`. A raised open palm then claims after only 0.5 s. |
| `follow`, `go_to` | FOLLOW |
| `stop`, `idle` | WAIT |
| `release_owner` | Releases the owner. |
| `circular` | Timed arc using `direction`/`duration`/`angular`. The angular speed is clamped, and the arc runs only when an owner is enrolled. `stop`, `follow` or `release` cancel it immediately. |

**Selecting the owner by click.** `/select_target` carries a click on the video from
`select.html`. Accepted formats are JSON `{x,y}`, `{u,v}`, `[x,y]` or `"x,y"`, in
pixels or normalised to 0..1. The clicked person is claimed at once, but only while
nobody owns the robot.

**Reported state.** `/robot_state` uses the bridge's vocabulary (IDLE, SEARCHING,
ENROLLING, OWNED, FOLLOWING, STOPPED), so its `_guidance()` table stays correct.

**Face.** `robot_face_ros.py` also listens to `/face_event`. `listening`, `speaking`
and `thinking` are shown for at most 6 s, or until an end event arrives; after that
the robot-state expression returns.

## 3f. Camera <-> lidar fusion (following around corners)

**What FollowMe publishes on the Jetson.** All points are in the camera optical frame,
stamped with the image time:

| Topic | Type | Content |
|---|---|---|
| `/followme/owner_position` | PointStamped | owner x, y, z from the camera, only on frames where the owner was actually detected |
| `/followme/owner_position_filtered` | PointStamped | Kalman-filtered owner position |
| `/followme/persons` | PoseArray | every detected person's 3-D centroid (computed only while someone subscribes) |

**The fusion node on the lidar PC:** `lidar_owner_fusion.py` (entry point `lidar_owner_fusion`).
1. It clusters `/scan`; the two legs of one person are merged into one cluster.
2. It moves the camera owner point into the laser frame with TF.
3. The lidar cluster next to that point is the owner.
4. That cluster keeps being tracked with the lidar alone, for up to 30 s after the last
   camera confirmation, including when the owner is outside the camera's 70° view.
5. It publishes `/owner_fused_position` and `/owner_fused_markers` for RViz: green
   means confirmed by the camera, orange means tracked by the lidar only.

**How FollowMe uses it.** When the camera's own track has expired, FollowMe follows
`/owner_fused_position` (if it is younger than 0.5 s). It transforms it into the camera
frame and keeps FOLLOW. This also counts as "owner seen", so there is no auto-release.

**TF requirements:** base_link -> laser (URDF), base_link -> camera_link (static transform
with the camera mount pose), and camera_link -> camera_color_optical_frame (RealSense driver).

**Driving safety:** the 0.9 m hard stop, the follow distance and the safety circle
all still apply.

## 3g. Steering direction (angular_sign / mirror_x)

If the base steers the wrong way (away from the owner instead of toward them), there
are two independent, genuinely live-settable ROS parameters on `/followme` — no
restart needed, changes apply on the very next frame:

* **`mirror_x`** (default `false`) — flips left/right **at the vision source**
  (right after 3-D deprojection). Use this if the camera itself reads mirrored:
  it keeps steering, search-turn direction, and goal placement all internally
  consistent, unlike a raw output-only flip.
* **`angular_sign`** (default `1.0`, only in `control_mode: cmd_vel`) — flips only
  the final `/cmd_vel` sign, for when the *drive base itself* is wired backwards
  (independent of anything vision-related).

```bash
ros2 param set /followme mirror_x true       # or angular_sign -1.0
```

Test methodically: change **one** parameter at a time, walk left/right in front of
the robot, and watch `/cmd_vel`'s `angular.z` sign plus the actual wheel direction.
Do not combine an unconfirmed `mirror_x` with an unconfirmed `angular_sign` —
each hides the effect of the other.

⚠️ **Single-instance guard.** `followme_node` now refuses to start if another
instance is already running (PID lockfile at `/tmp/followme_node.lock`, stale
locks from crashed processes are reclaimed automatically). Two nodes silently
racing on `/cmd_vel` under the same name was a real, previously-invisible bug:
whichever process's message landed last on a given cycle won the wheels, making
every `ros2 param get/set` and every physical steering test unreliable with no
error shown anywhere. If startup now logs `FATAL: another followme_node is
already running (pid N)`, that PID is genuinely still alive — `kill N` or run
`~/stop_a2rbot.sh` first.

⚠️ **Older builds only read `angular_sign` once at startup** — `ros2 param set`
would report success but silently change nothing. Both parameters are now backed
by a real parameter callback (`_on_param_change`); confirm with the log line
`angular_sign -> ... (live)` / `mirror_x -> ... (live)` after each `param set`.

## 4. Setup

```bash
pip install -e ".[followme]"          # or: pip install -r requirements.txt
pip install ultralytics               # segmentation
# MediaPipe >= 0.10.30 has only the Tasks API: download hand_landmarker.task
# (MediaPipe "Hand landmarker" model card) and set gestures.task_model_path.
```

### Re-ID weights (paper network)
```bash
python scripts/followme_train_mmt_pretrain.py --msmt17 /data/MSMT17_V1 \
       --out models/mmt_msmt17_ibn50a_256.pth.tar          # 80 epochs x 200 it, 1 GPU
python scripts/followme_export_mmt_onnx.py models/mmt_msmt17_ibn50a_256.pth.tar models/mmt_ibn50a.onnx
```

### Gesture SVM (Sect. VI-A)
```bash
python scripts/followme_collect_gestures.py --subject manuel --source realsense   # 1/2/3 record, Q save
python scripts/followme_train_gestures.py data/gestures/*.npz --protocol loso --out models/gesture_svm.joblib
```
If no model is found, gestures are disabled and the F/W keys or the
`~/follow` and `~/wait` services replace them.

## 5. Running

### Standalone (no ROS; goals in the camera-centred robot frame)
```bash
python followme_app.py --source realsense --config config/followme.yaml
```
1. Press **C** with the target *alone* in view and walking near and far for
   about 10 s (150 samples).
2. Press **S** to save `models/target.npz`, which is reloaded next time.
3. Press **F** or **W**, or use the gestures.

The overlay shows each person's Eq. 1 distance against λ_d, the target
position and θ, the state, and the gesture. Every frame is logged to
`logs/followme.csv`, including the per-stage latency.

### ROS 2 Humble + Nav2
```bash
cd ~/ros2_ws/src && ln -s ~/owner_tracker/ros2/followme_ros .
cd ~/ros2_ws && colcon build --packages-select followme_ros && source install/setup.bash
ros2 launch followme_ros followme.launch.py followme_root:=$HOME/owner_tracker \
     followme_config:=$HOME/owner_tracker/config/followme_a2rbot.yaml enable_navigation:=false
ros2 service call /followme/calibrate std_srvs/srv/Trigger     # target alone, walk near/far
ros2 service call /followme/save_target std_srvs/srv/Trigger
# watch /followme/debug_image, /followme/target_pose, /followme/safety_circle in RViz, then:
ros2 launch followme_ros followme.launch.py ... enable_navigation:=true
```

Requirements and behaviour:
* The node needs TF from `global_frame` (default `map`) to the colour optical
  frame, and Nav2's `navigate_to_pose` action.
* Other people are avoided because Nav2's costmap sees them in the laser
  scan, as in the paper.
* Without that TF the node never sends a goal: the target counts as not seen (!α).
* Keep Nav2's `max_vel_x` equal to `navigation.max_velocity_mps`; otherwise
  the Eq. 3 safety radius is wrong.

## 6. Reproducing the experiments

```bash
python scripts/followme_train_gestures.py data/gestures/*.npz --protocol split   # Fig. 6a, Table I left
python scripts/followme_eval_reid.py --root data/reid --reid-model models/mmt_msmt17_ibn50a_256.pth.tar \
       --out results/reid                                                            # Fig. 2, 6b, Table I right
```

The Re-ID dataset layout is documented at the top of `scripts/followme_eval_reid.py`.

Scoring:
* On the everyone-present set, selecting a *different* person counts as a
  miss in the matrix and is also reported separately (`wrong-person`).
  That is the error that matters for following.
* For timing, use the `total_ms` column of the app CSV; the paper reports
  7–10 Hz for 1–10 people.

## 7. Tests and verification status

```bash
python -m pytest -q        # 145 passed, 1 skipped (the PyTorch MMT test, torch not installed)
```

**What the tests cover:**
* **Paper maths:** Eq. 1, λ_d, the split, the ξ filter, Eq. 3, Kalman-filter expiration, deprojection.
* **Decision logic:** every Fig. 5 transition and the navigation executor.
* **Synthetic RGB-D end to end:**
  * calibration;
  * identification while a distractor crosses in front (no wrong-person selection);
  * FOLLOW, then SEARCH after t_exp, then STEADY after one full turn;
  * safety stop and WAIT.
* **ONNX Re-ID backend:** a real ONNX graph.
* **ROS 2 node:** message conversions, and the node's wiring (TF → map-frame
  goal, services, debug image) against stand-ins for rclpy.

**Not executed in the development environment:**
* **PyTorch parts:** the MMT network, its training and ONNX export (written
  to match the MMT repository's module and parameter names).
* **Models that need downloaded weights:** the Ultralytics segmenter and
  live MediaPipe.
* **The ROS 2 node on real ROS / Nav2.**

Validate each of these on the robot, in that order, with `enable_navigation:=false` first.
