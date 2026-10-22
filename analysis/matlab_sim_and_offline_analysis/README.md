# A2RBOT — MATLAB simulation and result plots

Run in MATLAB (R2019b or later; also tested in GNU Octave 8) from this folder.

| File | What it does | Data |
|---|---|---|
| `a2rbot_follow_sim.m` | Function: kinematic bicycle-model simulation of the deployed FollowMe goal-following law (`pi_goal`, `followme/direct_control.py`) with the parameters of `config/followme_deploy.yaml` (md5 112294b5…) | — |
| `run_follow_scenarios.m` | Owner walking away at 0.15 / 0.20 / 0.24 / 0.30 m/s; L-shaped walk at 0.10 s and 0.27 s update period. Saves `fig_sim_range_vs_speed.png`, `fig_sim_Lturn.png` | — (simulation) |
| `plot_results.m` | Plots the measured results. Saves `fig_reliability_vs_range.png`, `fig_heldout_rules.png`, `fig_acceptance_by_range.png`, `fig_latency_cdf.png`, `fig_sept_separation.png` | `data/*.csv` |

## Data files (all measured on the robot)

| File | Source |
|---|---|
| `reliability_by_range.csv` | Cue reliability (Youden's index) per owner-range band, fitted on block A + first half of the volunteer block, 10 Oct 2026 |
| `heldout_results.csv` | Five decision rules on the held-out frames (owner 1,110, volunteer 295) |
| `acceptance_by_range.csv` | Owner acceptance per range band, held-out frames |
| `timing_per_frame.csv` | Per-frame perception timings (ms) from the cue log, 15 W |
| `tegrastats_15W.csv` | CPU, GPU, input power, RAM, junction temperature, 15 W |
| `sept_session_separation.csv` | Sep 2026 box-based system: per-session separation, gated vs fixed (labels = the system's own decisions) |

## Expected console output of `run_follow_scenarios`

```
owner 0.15 m/s: range for t > 25 s  mean 1.302 m
owner 0.20 m/s: range for t > 25 s  mean 1.448 m
owner 0.24 m/s: range for t > 25 s  mean 2.007 m
owner 0.30 m/s: range for t > 25 s  mean 3.957 m   (range keeps growing)
minimum turning radius L/tan(delta_max) = 0.485 m
```

The simulation models the control law only (no perception errors beyond the optional
position noise, no wheel slip, camera assumed at the rear-axle origin). Label simulated
results as simulation in the report and presentation.
