# Follow controller simulation

A kinematic bicycle-model simulation of the goal-following law the robot uses (the `pi_goal` law in `jetson_owner_tracker/followme/direct_control.py`), with the same parameters as the deployed config: 0.28 m wheelbase, 30° steering, 0.24 m/s speed cap, acceleration and steering-rate limits.

- `a2rbot_follow_sim.m`: the simulation function (MATLAB / GNU Octave)
- `run_follow_scenarios.m`: owner walking straight away at different speeds, and an L-shaped walk with fast and slow perception updates
- `follow_sim.py`: the same idea in Python (numpy)

```matlab
>> run_follow_scenarios
```

It only models the control law: no perception errors (apart from optional position noise), no wheel slip, and the camera is assumed to sit on the rear axle.
