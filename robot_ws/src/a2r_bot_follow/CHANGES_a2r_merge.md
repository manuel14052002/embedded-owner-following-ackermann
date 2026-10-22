# a2r_merge — what changed (Oct 2026)

Back up the package on the Pi first, then replace it with this folder and rebuild:

    cd ~/a2r_bot_ws && colcon build --packages-select a2r_bot_follow && source install/setup.bash

## 1. maneuver.py is gone
The multi-point-turn planner now lives inside `smart_planner_node.py` (section
"MULTI-POINT TURNS"). Nothing else imported it. Delete `maneuver.py` from the Pi's
source tree too, or leave it; nothing uses it any more.

## 2. Reversing: shorter, owner-aware, measured
- K-turn plans: max 0.5 m per reverse leg and 1.2 m per plan
  (`maneuver_max_rev_leg_m`, `maneuver_max_reverse_m`).
- Owner keep-out zone of 0.75 m (`maneuver_owner_keep_m`). Reversing away from the owner costs extra
  (`maneuver_away_w`).
- Owner position comes from the lidar-leg track `/follow/owner` (`owner_topic`). If that is stale,
  it is estimated from the follow goal, which itself comes from the path, the owner, the phone or the last-seen position.
- While reversing, the leg stops early once the owner is ahead, or once the forward full-lock turn
  toward the live goal is clear (after `maneuver_rev_cut_min_m`).
- Progress on a steered leg is the larger of odometry and the IMU heading change. Slip or ESC lag
  can no longer make it over-reverse.
- Stall escape: reverse distances are 0.30, 0.45 and 0.60 m (was up to 1.2 m). The reverse ends on measured
  distance, with the timer kept only as a backstop. The chosen lock prefers swinging the nose toward the owner.
- Face-owner reverse is capped at 0.45 m (`goal_face_max_m`).
- Search is about 2x faster (coarser bins, weighted A*). The time budget is 0.8 s instead of 2.0 s.

## 3. U-turn lag
The "is the full-lock U-turn blocked?" test used the scan that still contains the owner. The owner's
own legs made it start a K-turn (stand still, plan, shuffle). Points within 0.45 m of the owner are
now ignored for that test only. The collision monitor still stops for the owner.

## 4. Walls / owner lost
- follow_manager: when the Jetson planner runs but there is no path, no camera owner and no phone
  fix, it now SEARCHES. It goes to the last-seen position (lidar legs, phone fix, or the end of the
  Jetson path, kept in odom), then sweeps, roams and repeats until the owner is tracked again
  (`search_enabled: true`, `search_with_planner: true`, start after `owner_timeout_s + search_start_s`).
  Before this change it published no goal and the car stood still.
- smart_planner: owner lost and a wall closer than 0.45 m in front: back off 0.35 m on the lock that
  swings the nose toward the side the owner was last seen (`lost_backoff_*`).
- The "reverse gains little - waiting" stare dropped from 1.5 s to 0.6 s (`escape_force_after_s`),
  and the reverse cooldown from 2.0 s to 1.2 s.
- NOTE: `NO_OWNER` is still in `hold_states`. If FollowMe reports NO_OWNER when it loses you, the car
  holds instead of searching. Remove it from `hold_states` if that is what you see.

## 5. CPU
- RViz no longer starts by default (`use_rviz:=true` to get it). Run RViz on a laptop with the same
  ROS_DOMAIN_ID; on the Pi it costs most of a core.
- Command chain 50 Hz -> 25 Hz (`cmd_rate_hz`). Collision-monitor state JSON is published at 10 Hz plus on
  every change, and is parsed once in the planner (was twice per message at 50 Hz).
- owner_fusion in Jetson mode no longer subscribes to the raw 2300-beam /scan and to odometry just
  to get a tick; it uses a 10 Hz timer.
- scan_leg_fill no longer copies intensities into /scan/planner and /scan/merged, which halves the
  bytes every subscriber deserialises.
- follow_manager goal rate 20 -> 10 Hz. The routing histogram and ray tests are vectorised
  (identical results).
- Planner and monitor cache beam cos/sin tables; manoeuvre obstacle points are built once per scan.
- Planner rate 5 -> 7 Hz: one plan per lidar sweep, for responsiveness. It is cheap.

Measure on the Pi with `top` (press `1` for per-core) or `ps -eo pcpu,comm --sort=-pcpu | head`.

## 6. Fixes from the Pi-spec simulation (second round)
Validated in a closed-loop simulation of the real node code. Every callback was timed and scaled to Pi 4
speed (6x, and 9x for a throttled Pi). The car model included ESC stall and reverse handshake, servo lag,
12 % odometry slip, IMU bias and a 7 Hz lidar. There were 8 scenarios x 3 seeds. No collisions occurred.
Reverse distance was <= 0.72 m and the car stayed >= 0.78 m from the owner. A lost owner was found
(L-corner 11 s, behind a wall 5 s). Changes:
- `ackermann.first_hit` checks poses in chunks against nearby points only and stops at the first
  hit: same result on 3000 random tests, 6.5x faster. It was the planner's main CPU cost in corridors.
- K-turn legs are at least 0.2 m (`maneuver_min_leg_m`). A trailing stub leg is skipped and follow takes over.
- Only reverse legs are split at `maneuver_max_leg_m`. Forward legs run in one go.
- After a reverse is cut short (forward turn free), the car does not face-reverse again for 1.5 s.
- The owner more than 45 deg round at the stand-off -> owner-aware K-turn (`maneuver_face_deg` 45, was 75).
  `goal_face_cooldown_s` 1.0 (was 3.0: the car stood 3 s with the owner beside it).
- Every node calls `gc.freeze()` after start-up. Full garbage-collector passes were 20-35 ms stalls,
  ~200 ms on the Pi, at random moments.
- smart_planner uses rclpy's SingleThreadedExecutor (`A2R_PLANNER_MT=1` brings the 3-thread one back).
Known trade-off: U-turns in 1.2-1.3 m corridors take 5-7 s longer than before. The old code was faster
because it backed up to 2.4 m, to within 0.43 m of the owner. Wall clearance in those corridors gets down
to 6-7 cm; raise `maneuver_margin_m` to 0.06 if that looks too tight.

## Removing the old .bak files on the Pi
    find ~/a2r_bot_ws/src/a2r_bot_follow \( -name '*.bak*' -o -name '*_bak' -o -name '__pycache__' \) -prune -exec rm -rf {} +
    rm -f ~/a2r_bot_ws/src/a2r_bot_follow/a2r_bot_follow/maneuver.py
