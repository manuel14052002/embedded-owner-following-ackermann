# Sensor fusion

Three nodes, one shared maths library (`kalman.py`: linear KF, EKF,
Madgwick/Mahony attitude — plain numpy, no filterpy, no scipy).

## 1. `imu_filter_node` — attitude

`/imu/data_raw` + `/imu/mag` → `/imu/data`, now with an orientation.

The firmware deliberately ships a zero quaternion.  An 8-bit-era
microcontroller budget is better spent on deterministic PWM timing than on
quaternion algebra, and the Pi has floating point to spare.

**Why Madgwick and not a Kalman filter over the quaternion.**  Attitude lives
on a manifold, not in ℝⁿ.  A quaternion has four components and three degrees
of freedom, so an unconstrained Kalman filter over its components drifts off
the unit sphere and you spend your life renormalising and fighting the
covariance.  Madgwick's gradient-descent formulation handles the constraint
natively and costs about 280 flops per update.

`beta` is the one knob: how fast the accelerometer is allowed to correct gyro
drift.  Too low and heading drifts; too high and every bump tips the estimate.
0.06 suits a car.

**Turn the magnetometer off** if the GY-91 sits near the motor or the ESC
leads.  A compass that follows the throttle is worse than no compass, because
the EKF believes it.  When it is off, the node inflates the published yaw
variance 50x — yaw is then genuinely unobservable, and saying otherwise would
be lying to the filter downstream.

## 2. `ekf_state_node` — pose

Six states: `[px, py, θ, v, ω, a]`.

**Why acceleration is in the state.**  The car has no trusted drive encoder,
so forward speed is only weakly observed.  The accelerometer observes its
derivative directly.  Carrying `a` lets the filter integrate the accelerometer
into the speed estimate instead of discarding it, and gives it something
sensible to do in the half-second after a stop command, when the commanded
velocity says zero and the car is very much still moving.

`a` decays towards zero with a time constant, because a constant-acceleration
model applied to a robot that has stopped accelerating will happily predict it
accelerating forever.

Measurements fold in **separately** as each arrives, rather than as one
stacked observation.  A stacked `H` means the filter stalls whenever any one
sensor is late, and these sensors run at very different rates.

| Source | Observes | R |
|---|---|---|
| IMU yaw | θ | 2.5e-3 |
| **IMU yaw rate** | ω | **4.0e-4** |
| IMU accel (gravity-compensated) | a | 2.0e-1 |
| Commanded velocity | v | 4.0e-2 |
| Scan match | px, py, θ | from the match score |

Note the 100x asymmetry between the gyro's yaw rate and the command.  It is
deliberate: **the gyro measures what happened; the command only states what
was asked for.** On a slipping tyre or a stalled servo those are very
different things.

Two details that matter more than they look:

- **Gravity compensation on the forward accelerometer.** The car pitches under
  braking and on ramps, and gravity then leaks into `x` at `g·sin(pitch)` —
  1.7 m/s² on a 10° slope, more than the car can actually produce.
- **Zero-velocity update.** When the gyro is quiet and no command is active,
  the car is stationary. Saying so explicitly collapses the velocity variance
  and stops the position wandering while parked. It is the cheapest
  improvement available to any dead-reckoning filter.

Every update is **chi-square gated**.  A lidar ray clipping a passing leg, a
magnetometer sample taken as the motor kicks in — each is individually
plausible and collectively poisonous. Gating discards them without anyone
having to enumerate the failure modes in advance.

Covariances use **Joseph form**.  The textbook `P = (I − KH)P⁻` is
algebraically identical but loses symmetry after a few thousand float
iterations, and a filter whose covariance has gone indefinite produces
confident nonsense.

## 3. `scan_fusion_node` — lidar + sonar

`/scan` + `/ultrasonic/front` + `/ultrasonic/rear` → `/scan/fused`.

### Why, when the lidar already sees everything

It does not, and the gaps are the ones that hurt.

- **Glass, gloss and dark fabric.** A 2-D lidar fires infrared and waits. A
  glass door returns almost nothing; matte black trousers return very little.
  Both read as infinite range, which the planner reads as free space, and
  drives into. Ultrasound does not care about colour or IR reflectivity — it
  cares about acoustic impedance, and glass is an excellent acoustic reflector.
- **The blind cone below the scan plane.** The lidar sweeps a horizontal plane
  from the top deck. A kerb, a doorstep, a cat, the foot of a table — all
  invisible. The front sonar is mounted low on the nose.
- **Behind.** The lidar is occluded to the rear by the battery and the Jetson.
  The escape manoeuvre reverses, so this matters.

Ultrasound has its own failures: a 60° beam that cannot say *where* in the
cone, specular reflection off angled surfaces, crosstalk. So neither sensor is
trusted alone.

### How

The scan is divided into 72 sectors of 5°, each carrying a 1-D Kalman filter
over "range to the nearest obstacle in this sector".  Vectorised in numpy —
72 Python-level filter updates would cost 14 ms on a Pi; four array operations
cost 0.4 ms.

Two rules make this fusion rather than averaging:

1. **The measurement models differ.** A lidar beam measures range at a precise
   bearing. A sonar measures range to the nearest thing *anywhere* in a 60°
   cone. So a sonar reading is applied to every sector in its cone, but only
   as an **upper bound** — strong evidence that no sector ahead is clear
   beyond 0.8 m, and no evidence at all about which sector it is.

2. **Disagreement resolves to the pessimist, asymmetrically.** Lidar says 5 m,
   sonar says 0.9 m → 0.9 m (the glass door). Lidar says 0.9 m, sonar says
   5 m → still 0.9 m (a sonar missing a table leg is routine). The
   `sonar_only_shortens` rule encodes this. **Leave it on.** With it off, a
   sonar that misses a table leg will erase the lidar's detection of it.

Both cases are verified in the test output: glass door (lidar `inf`, sonar
0.85 m → fused 0.85 m) and table leg (lidar 0.55 m, sonar 4.2 m → fused
0.55 m).

The prediction step propagates each sector's range forward by the robot's own
motion.  At 1.2 m/s the car covers 12 cm between scans, so a sector that is
briefly occluded decays towards clear at the rate the geometry allows rather
than snapping back to "free".
