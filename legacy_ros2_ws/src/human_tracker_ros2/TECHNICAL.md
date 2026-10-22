# Technical Documentation

## System Overview

This document provides detailed technical information about the human tracking system's implementation, algorithms, and design decisions.

## Architecture

### Component Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    Tracker Node                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │           State Machine                            │  │
│  │  ┌──────────────┐      ┌──────────────┐           │  │
│  │  │  WAITING_FOR │─────>│    TARGET    │           │  │
│  │  │   TARGET     │<─────│    LOCKED    │           │  │
│  │  └──────────────┘      └──────────────┘           │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌─────────────────┐        ┌─────────────────┐         │
│  │  Gesture        │        │  Person         │         │
│  │  Detector       │        │  Tracker        │         │
│  │  (MediaPipe)    │        │  (MediaPipe)    │         │
│  └─────────────────┘        └─────────────────┘         │
└──────────────────────────────────────────────────────────┘
                    ▲                    │
                    │                    ▼
            ┌───────────────┐    ┌──────────────┐
            │  Camera Node  │    │ Robot Base   │
            │  (RealSense)  │    │ (cmd_vel)    │
            └───────────────┘    └──────────────┘
```

## Algorithm Details

### 1. Hand Gesture Recognition

#### MediaPipe Hands Solution

The system uses Google's MediaPipe Hands, which provides:
- 21 hand landmarks per detected hand
- Real-time tracking at 30+ FPS
- Multi-hand support (up to 2 hands)

#### Raised Hand Detection

**Algorithm:**
```python
def is_hand_raised(hand_landmarks):
    wrist = hand_landmarks[WRIST]
    middle_finger_mcp = hand_landmarks[MIDDLE_FINGER_MCP]
    
    # Hand is raised if middle finger base is above wrist
    threshold = 0.1  # normalized coordinate space
    return middle_finger_mcp.y < wrist.y - threshold
```

**Rationale:**
- Y-coordinates in MediaPipe are normalized (0 to 1)
- Lower Y values indicate higher position in frame
- Threshold prevents false positives from small movements

#### Extended Finger Counting

**Algorithm:**
```python
def count_extended_fingers(hand_landmarks):
    count = 0
    
    # Thumb: Check horizontal displacement
    if abs(thumb_tip.x - thumb_ip.x) > 0.05:
        count += 1
    
    # Other fingers: Compare tip Y with PIP joint Y
    for tip, pip in finger_pairs:
        if tip.y < pip.y:  # Tip above PIP = extended
            count += 1
    
    return count
```

**Challenges:**
1. **Hand orientation**: Thumb detection varies with hand rotation
   - Solution: Use X-coordinate difference for thumb
   
2. **Partial occlusion**: Fingers may be partially hidden
   - Solution: MediaPipe's tracking handles this well
   
3. **False positives**: Hand movements can trigger counts
   - Solution: Temporal smoothing with 2-second hold requirement

#### Temporal Filtering (2-Second Hold)

**Algorithm:**
```python
class TemporalFilter:
    def __init__(self, duration=2.0):
        self.required_duration = duration
        self.start_time = None
    
    def update(self, gesture_detected):
        if gesture_detected:
            if self.start_time is None:
                self.start_time = time.time()
            elapsed = time.time() - self.start_time
            return elapsed >= self.required_duration
        else:
            self.start_time = None
            return False
```

**Benefits:**
- Eliminates false positives from quick movements
- Provides clear user feedback (countdown display)
- Intentional gesture requirement

### 2. Person Detection and Tracking

#### MediaPipe Pose Solution

Features:
- 33 body landmarks (full body)
- 3D landmark coordinates (X, Y, Z)
- Model complexity options (0, 1, 2)

#### Pose-Based Position Estimation

**2D Position (Image Coordinates):**
```python
def compute_center_2d(pose_landmarks):
    # Average of shoulder and hip landmarks
    keypoints = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    center_x = mean([lm.x for lm in keypoints]) * image_width
    center_y = mean([lm.y for lm in keypoints]) * image_height
    return (center_x, center_y)
```

**3D Position (Camera Coordinates):**
```python
def compute_position_3d(center_2d, depth_frame, camera_intrinsics):
    x_pixel, y_pixel = center_2d
    depth = depth_frame[y_pixel, x_pixel] / 1000.0  # mm to meters
    
    # Perspective projection
    x = (x_pixel - cx) * depth / fx
    y = (y_pixel - cy) * depth / fy
    z = depth
    
    return (x, y, z)
```

**Camera Intrinsics:**
- fx, fy: Focal lengths (pixels)
- cx, cy: Principal point (image center)

#### Orientation Estimation

**Yaw Calculation:**
```python
def compute_orientation(pose_landmarks):
    left_shoulder = pose_landmarks[LEFT_SHOULDER]
    right_shoulder = pose_landmarks[RIGHT_SHOULDER]
    
    # Vector from left to right shoulder
    dx = right_shoulder.x - left_shoulder.x
    dy = right_shoulder.y - left_shoulder.y
    
    # Arctangent gives angle
    yaw = arctan2(dy, dx)
    return yaw
```

**Limitations:**
- Only yaw (rotation around vertical axis) is computed
- Pitch and roll could be added using more landmarks
- Assumes upright posture

#### Bounding Box Calculation

**Algorithm:**
```python
def compute_bbox(pose_landmarks):
    x_coords = [lm.x * width for lm in landmarks]
    y_coords = [lm.y * height for lm in landmarks]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Add padding for visibility
    padding = 20
    return (x_min - padding, y_min - padding, 
            x_max - x_min + 2*padding, 
            y_max - y_min + 2*padding)
```

### 3. Target Following Control

#### Proportional Controller

**Control Law:**
```
linear_velocity = Kp_linear * distance_error
angular_velocity = Kp_angular * angular_error
```

Where:
- `distance_error = current_distance - target_distance`
- `angular_error = arctan2(x_offset, depth)`

**Tuning Parameters:**
- Kp_linear = 0.5 (slower for safety)
- Kp_angular = 1.0 (responsive turning)

**Dead Zones:**
```python
if abs(distance_error) < 0.2:  # 20cm
    linear_velocity = 0
    
if abs(angular_error) < 0.1:   # ~5.7 degrees
    angular_velocity = 0
```

Benefits:
- Prevents oscillation near target
- Reduces motor wear
- Smoother motion

#### Velocity Clamping

```python
v_linear = clip(v_linear, -v_max, v_max)
v_angular = clip(v_angular, -w_max, w_max)
```

Safety limits:
- Max linear: 0.5 m/s (walking speed)
- Max angular: 1.0 rad/s (~57 deg/s)

### 4. State Machine

```
┌──────────────────┐
│ WAITING_FOR_     │
│    TARGET        │
│                  │
│ - Detect gestures│
│ - No tracking    │
└────────┬─────────┘
         │ Hand raised 2s
         │
         v
┌──────────────────┐
│   TARGET_        │
│    LOCKED        │
│                  │
│ - Track person   │
│ - Follow target  │
│ - Check for stop │
└────────┬─────────┘
         │ Stop gesture
         │
         v
┌──────────────────┐
│   WAITING_FOR_   │
│     TARGET       │
└──────────────────┘
```

**State Transitions:**

1. **WAITING → LOCKED:**
   - Condition: Hand raised AND held for 2 seconds
   - Action: Lock first detected person, start tracking

2. **LOCKED → WAITING:**
   - Condition: Stop sign detected (5 fingers)
   - Action: Release target, reset trackers, stop robot

**Target Loss Handling:**
- If person not detected for N frames, show warning
- Continue searching in last known position
- Don't automatically unlock (requires stop gesture)

## Performance Optimization

### 1. MediaPipe Optimization

**Model Complexity Selection:**
```
Model 0: Fastest, least accurate (~20ms/frame on Jetson)
Model 1: Balanced (default, ~40ms/frame)
Model 2: Most accurate, slowest (~80ms/frame)
```

**Thread Configuration:**
```python
# MediaPipe automatically uses available cores
# On Jetson Nano (4 cores), typically uses 2-3 cores
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7
)
```

### 2. Image Processing Pipeline

**Processing Flow:**
```
RealSense → ROS Image → CV Bridge → NumPy Array → MediaPipe → Results
```

**Optimizations:**
- Direct memory access with NumPy
- Vectorized operations
- Minimal copying (use views where possible)

**Memory Layout:**
```python
# Efficient BGR to RGB conversion
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # In-place when possible
```

### 3. ROS 2 Communication

**Queue Sizes:**
- Image topics: 10 (balance latency vs. memory)
- Command topics: 10 (recent commands important)

**Publishing Strategy:**
```python
# Publish only when needed
if target_locked and position_changed:
    pub.publish(msg)
```

### 4. Jetson-Specific Optimizations

**CPU Governor:**
```bash
# Set to performance mode
sudo nvpmodel -m 0  # Maximum performance mode
sudo jetson_clocks  # Lock clocks to maximum
```

**Memory:**
```bash
# Disable swap for real-time performance
sudo swapoff -a
```

**Power:**
```bash
# Monitor power consumption
sudo tegrastats
```

## Challenges and Solutions

### Challenge 1: Latency in Depth-RGB Alignment

**Problem:**
- Depth and RGB sensors have different framerates
- Alignment introduces 1-2 frame delay

**Solutions:**
1. **Hardware timestamping**: Use RealSense timestamps
2. **Alignment in camera node**: Process before publishing
3. **Predictive filtering**: Estimate position at current time

**Implementation:**
```python
# Align depth to color in camera
align = rs.align(rs.stream.color)
aligned_frames = align.process(frames)
```

### Challenge 2: False Gesture Detection

**Problem:**
- Random hand movements trigger locks
- Natural gestures (waving) detected as intentional

**Solutions:**
1. **Temporal filtering**: Require 2-second hold
2. **Confidence thresholding**: Set min_detection_confidence
3. **State machine**: Prevent rapid state changes

**Configuration:**
```yaml
gesture_detector:
  min_detection_confidence: 0.7  # Higher = fewer false positives
  hand_raise_duration: 2.0       # Longer = more intentional
```

### Challenge 3: Person Re-identification

**Problem:**
- MediaPipe Pose doesn't maintain identity
- If person leaves and returns, treated as new

**Current Limitation:**
- System doesn't re-identify same person
- Requires stop gesture to unlock

**Potential Solutions:**
1. **Feature matching**: Compare pose features
2. **Appearance models**: Use clothing color/pattern
3. **Face recognition**: Add face ID (privacy concerns)

### Challenge 4: Multi-Person Scenarios

**Problem:**
- Multiple people in frame
- Which person to track?

**Current Approach:**
- Lock first person who raises hand
- Ignore others until unlock

**Future Improvements:**
```python
class MultiTargetTracker:
    def __init__(self):
        self.targets = []
        self.active_target = None
    
    def select_target(self, candidates):
        # Priority: first to raise hand
        # Alternative: closest, facing camera, etc.
        pass
```

### Challenge 5: Occlusion Handling

**Problem:**
- Target walks behind obstacle
- Pose landmarks lost temporarily

**Current Behavior:**
- Show "TARGET LOST" warning
- Continue attempting to reacquire
- Maintain last known position

**Improvements:**
```python
class OcclusionHandler:
    def __init__(self, max_lost_frames=30):
        self.lost_count = 0
        self.last_position = None
    
    def update(self, detection):
        if detection is None:
            self.lost_count += 1
            if self.lost_count > max_lost_frames:
                # Predict position or search pattern
                return self.search_strategy()
        else:
            self.lost_count = 0
            self.last_position = detection['position']
```

### Challenge 6: Lighting Conditions

**Problem:**
- Poor lighting affects detection
- Glare/shadows cause issues

**Mitigation:**
```python
# Preprocessing for challenging conditions
def preprocess_image(image):
    # CLAHE for contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
```

**Best Practices:**
- Ensure even, diffuse lighting
- Avoid backlighting
- Recommended illuminance: 300-500 lux

## Coordinate Systems

### Camera Coordinate System (RealSense)

```
       Z (depth, forward)
      /
     /
    ──────> X (right)
    |
    |
    v Y (down)
```

### Robot Coordinate System (ROS REP-103)

```
       X (forward)
      /
     /
    ──────> Y (left)
    |
    |
    v Z (up)
```

**Transformation:**
```python
# Camera to Robot transformation
robot_x = camera_z
robot_y = -camera_x
robot_z = -camera_y
```

## Testing and Validation

### Unit Tests

Test individual components:
```bash
# Test gesture detector
python3 -m pytest tests/test_gesture_detector.py

# Test person tracker
python3 -m pytest tests/test_person_tracker.py
```

### Integration Tests

Test ROS 2 nodes:
```bash
# Test camera node
ros2 run human_tracker_ros2 camera_node

# Verify topics
ros2 topic list
ros2 topic hz /camera/color/image_raw
```

### Performance Benchmarks

```python
# Measure processing time
import time

start = time.time()
result = gesture_detector.detect_gestures(frame)
elapsed = time.time() - start

print(f"Gesture detection: {elapsed*1000:.1f}ms")
# Target: <30ms for 30 FPS
```

## Future Enhancements

### 1. Advanced Tracking

- **Kalman filtering**: Smooth position estimates
- **Predictive tracking**: Anticipate motion
- **Multiple hypothesis tracking**: Handle ambiguity

### 2. Gesture Vocabulary

```python
gestures = {
    'stop': lambda: count_fingers() == 5,
    'follow': lambda: hand_raised() and count_fingers() == 1,
    'faster': lambda: two_hands_raised(),
    'slower': lambda: hand_lowering_gesture(),
}
```

### 3. Navigation Integration

```python
# Integration with Nav2
from nav2_simple_commander import BasicNavigator

nav = BasicNavigator()
nav.followPath(target_path)
```

### 4. Safety Features

- **Collision avoidance**: Use depth for obstacle detection
- **Emergency stop**: Immediate stop on command
- **Safe zones**: Define areas to avoid

### 5. Learning Capabilities

- **Gesture learning**: Train custom gestures
- **Behavior adaptation**: Learn user preferences
- **Environment mapping**: Build spatial memory

## References

- [MediaPipe Documentation](https://google.github.io/mediapipe/)
- [Intel RealSense SDK](https://dev.intelrealsense.com/)
- [ROS 2 Documentation](https://docs.ros.org/)
- [Jetson Optimization Guide](https://developer.nvidia.com/embedded/learn/tutorials)

## Appendix: Troubleshooting Checklist

- [ ] Camera connected and detected (`lsusb`)
- [ ] Depth stream enabled and aligned
- [ ] MediaPipe models installed
- [ ] ROS 2 environment sourced
- [ ] Sufficient lighting (300+ lux)
- [ ] No USB bandwidth issues
- [ ] Correct coordinate transformations
- [ ] Gesture thresholds tuned
- [ ] Performance mode enabled (Jetson)
- [ ] No competing processes
