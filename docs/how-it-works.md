# How it works

This is the longer version of the overview in the main README. I've tried to explain the ideas rather than every parameter; the actual values are in the config files.

## 1. Seeing people (Jetson)

Every camera frame goes through the same steps:

1. **Segmentation.** YOLO11n-seg (TensorRT, 480 px input) finds people and gives a mask for each one.
2. **Embedding.** Each person crop goes through OSNet x1_0, which turns it into a 512-number "appearance fingerprint".
3. **Identification.** The fingerprint is compared with what the robot knows about its owner (next section).
4. **Localisation.** For the owner, I take the depth pixels inside the (slightly eroded) mask and average them to get a 3D position, then smooth it with a small Kalman filter.
5. **Gestures.** MediaPipe finds the owner's hand landmarks and a small SVM classifies them as follow / wait / other.

This is the same structure as FollowMe. The changes were mostly about making it fit on a Jetson: YOLO11n-seg instead of YOLACT++, OSNet instead of a ResNet-50 re-ID model, and everything exported to TensorRT FP16.

## 2. Learning and recognising the owner

**Enrolment.** When nobody owns the robot, it waits for someone to hold up an open palm for about 1.5 s. Then it spends 12 s collecting 120 crops of that person, but only from frames where they are alone (if someone overlaps them, the frame is skipped). Anyone else who walks past during this time is saved as a "stranger".

**Multi-view gallery.** Most re-ID systems average all the crops into one prototype. That works badly on a robot, because it sees you from the front, from behind, from the side, from 1 m and from 4 m, and those look very different. So I keep the crops as separate views:

- 80 of the 120 crops become the gallery;
- the other 40 are used to set the acceptance threshold;
- while following, new confident views are added (at most one per second, and only if they are not near-duplicates), up to 400.

To check a new person, the robot takes the three gallery views most similar to them and averages those similarities. The distance is `d = 1 - (average of the top-3 cosine similarities)`. The threshold `λ` is the mean distance of the 40 held-out crops plus 2.5 standard deviations. On top of that, the candidate must look more like the owner than like any known stranger, by a small margin.

**Accepting weaker evidence.** A strict `d ≤ λ` rule loses the owner too often (bad light, far away, partly hidden). So the decision is a cascade that accepts progressively weaker evidence, but each step needs something else to agree:

1. strict: `d ≤ λ` and the stranger check;
2. continuity: the box overlaps the owner's box from the previous frame and nobody else is close, `d ≤ 1.8λ`;
3. relaxed: inside the predicted position of the owner, `d ≤ 1.3λ`;
4. far away (> 3 m): a slightly looser threshold, but it must hold for 3 frames and agree with the lidar or the phone;
5. lidar: the only person within 0.8 m of where the lidar keeper says the owner is, `d ≤ 2λ`.

When two people cross, only the strict rule is allowed for a couple of seconds. That's the moment where trackers usually swap identities.

## 3. Keeping the owner with the lidar

The camera only sees about 70° in front of the robot. When you turn a corner or walk past the side of the car, you are gone from the image long before the car manages to turn. The lidar sees all around, so I wrote a tracker that keeps hold of the owner in the scan.

Each scan (7 Hz) is processed like this:

- points closer than 0.45 m (the robot itself) are dropped, and ICP between scans gives a rough odometry;
- the scan is split into clusters wherever there's a gap of more than ~12 cm;
- clusters are labelled by size: thin posts, legs (≤ 25 cm), pairs of legs, wider "blobs" (a body or a coat), and furniture (several thin legs close together, like chairs and tables);
- every person-like cluster gets its own track, matched from scan to scan by nearest neighbour with a gate that grows with speed.

The lidar can't tell people apart, so it doesn't decide who the owner is: **the camera does.** Whenever the camera sees the owner, the closest person track becomes the owner track. Between camera sightings the keeper simply follows that track. If the owner track merges with someone else, it publishes its prediction instead of a possibly wrong measurement. If the camera sees a stranger standing right on the owner track for a while, the track is dropped.

The keeper and the camera help each other in both directions:

- the keeper's position gives the camera a small search area (0.8 m), which is what step 5 of the cascade uses;
- the keeper's position is what the Raspberry Pi drives towards most of the time, with the camera position as a fallback and a sanity check (they must agree within 0.7 m).

There's also an experimental leg-shape signature (leg width and spacing) in `leg_signature.py` that could let the lidar recognise the owner on its own, but for now it only logs.

## 4. Driving (Raspberry Pi)

**Owner fusion and filtering.** `owner_fusion` takes the keeper point (or the camera point if the keeper is quiet) and passes one owner position to `follow_manager`. There it goes into a constant-velocity Kalman filter in the odometry frame. Measurements that don't fit (chi-square test, or a jump of more than 0.7 m) are rejected, which stops the filter from jumping to a bystander. If no measurement arrives, the filter predicts for up to 2 s before the owner counts as lost.

**Goal.** The goal is a point on the line to the owner, 1.6 m short of them. Speed is proportional to the remaining distance (capped at 0.24 m/s), and steering uses pure pursuit:

```
v = min(0.24, 0.8 * (d - 1.6)) * cos(bearing)
curvature = 2 * sin(bearing) / max(d, 1.2)
steering angle = atan(wheelbase * curvature)
```

The car has a 0.28 m wheelbase and 30° maximum steering, so the tightest turn is a circle of about 0.49 m radius. If the owner is more than 75° to the side, it just drives a full-lock arc.

**Obstacles.** If the straight path to the goal is blocked, the planner first tries the smallest bend that is free, and if that fails it hands over to one of the local planners (DWA, VFH+, APF or plain pure pursuit, selectable in the config). When it gets boxed in it reverses out.

**Safety.** Two layers sit between the planner and the motors and can't be bypassed by it:

- a command multiplexer with priorities (manual > escape manoeuvre > navigation);
- a collision monitor that checks the car's footprint against the raw lidar scan. It slows down or stops depending on speed: at full speed it stops about 0.31 m before an obstacle and starts slowing at about 1.06 m.

**Low level.** The serial bridge sends `{"cmd":"DRIVE","v":..,"s":..}` lines to the ESP32. The firmware maps them to servo and ESC pulses through a PCA9685, reads the IMU, and goes to neutral if no command arrives for 500 ms. There are no wheel encoders, so speed is open loop. The ESC only moves the car reliably above ~0.17 m/s, which is why the bridge keeps any non-zero command inside a narrow throttle band.

## 5. Simulation

`tools/simulation/` has a kinematic bicycle-model simulation of the goal law (MATLAB and a Python version). I used it to check what happens when the owner walks faster than the robot can drive, and how much perception delay the loop can tolerate before it starts to oscillate.
