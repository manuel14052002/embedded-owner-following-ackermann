# Usage Examples

This document provides various usage examples for the Human Tracker ROS 2 package.

## Table of Contents
1. [Basic Usage](#basic-usage)
2. [Advanced Scenarios](#advanced-scenarios)
3. [Integration Examples](#integration-examples)
4. [Troubleshooting Examples](#troubleshooting-examples)

---

## Basic Usage

### Example 1: Simple Tracking

Launch the system with default parameters:

```bash
# Terminal 1: Launch the tracker
ros2 launch human_tracker_ros2 tracker_launch.py

# Terminal 2: View the annotated video feed
ros2 run rqt_image_view rqt_image_view /tracker/annotated_image
```

**Workflow:**
1. Stand in front of camera (1-3 meters away)
2. Raise your hand
3. Keep it raised for 2 seconds
4. You'll see "TARGET LOCKED" on screen
5. Walk around - the robot will follow you
6. To stop: raise all 5 fingers (stop sign)

---

### Example 2: Custom Following Distance

Keep the robot further away:

```bash
ros2 launch human_tracker_ros2 tracker_launch.py \
    target_follow_distance:=2.5
```

Or closer:

```bash
ros2 launch human_tracker_ros2 tracker_launch.py \
    target_follow_distance:=1.0
```

---

### Example 3: Lower Resolution for Better Performance

```bash
ros2 launch human_tracker_ros2 tracker_launch.py \
    rgb_width:=424 \
    rgb_height:=240 \
    frame_rate:=15
```

---

## Advanced Scenarios

### Example 4: Running Nodes Separately

**Terminal 1: Camera Node**
```bash
ros2 run human_tracker_ros2 camera_node \
    --ros-args \
    -p frame_rate:=30 \
    -p rgb_width:=640 \
    -p rgb_height:=480
```

**Terminal 2: Tracker Node**
```bash
ros2 run human_tracker_ros2 tracker_node \
    --ros-args \
    -p target_follow_distance:=1.5 \
    -p max_linear_speed:=0.3
```

---

### Example 5: Recording Data

Record tracking session for analysis:

```bash
# Start recording
ros2 bag record -a -o tracking_session

# In another terminal, run the tracker
ros2 launch human_tracker_ros2 tracker_launch.py

# Stop recording with Ctrl+C
```

Playback recorded session:

```bash
ros2 bag play tracking_session
```

---

### Example 6: Monitoring Performance

**Terminal 1: Run tracker**
```bash
ros2 launch human_tracker_ros2 tracker_launch.py
```

**Terminal 2: Monitor topics**
```bash
# Check frame rates
ros2 topic hz /camera/color/image_raw
ros2 topic hz /tracker/annotated_image

# Monitor target position
ros2 topic echo /tracker/target_pose

# Monitor velocity commands
ros2 topic echo /cmd_vel
```

**Terminal 3: System resources**
```bash
# On Jetson
sudo tegrastats

# On other systems
htop
```

---

### Example 7: Custom Parameters from Config File

Create custom config file `my_config.yaml`:

```yaml
/**:
  ros__parameters:
    # Camera settings
    frame_rate: 30
    rgb_width: 640
    rgb_height: 480
    
    # Tracker settings
    target_follow_distance: 1.8
    max_linear_speed: 0.4
    max_angular_speed: 0.8
```

Launch with custom config:

```bash
ros2 launch human_tracker_ros2 tracker_launch.py \
    --ros-args --params-file my_config.yaml
```

---

## Integration Examples

### Example 8: Integration with Navigation Stack

```python
#!/usr/bin/env python3
"""
Example: Integrate tracker with Nav2
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool
from nav2_simple_commander.robot_navigator import BasicNavigator

class TrackerNavIntegration(Node):
    def __init__(self):
        super().__init__('tracker_nav_integration')
        
        self.navigator = BasicNavigator()
        
        # Subscribe to tracker outputs
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/tracker/target_pose',
            self.target_pose_callback,
            10
        )
        
        self.locked_sub = self.create_subscription(
            Bool,
            '/tracker/target_locked',
            self.locked_callback,
            10
        )
        
        self.target_locked = False
    
    def target_pose_callback(self, msg):
        if self.target_locked:
            # Convert tracker pose to navigation goal
            goal = self.convert_to_nav_goal(msg)
            self.navigator.goToPose(goal)
    
    def locked_callback(self, msg):
        self.target_locked = msg.data
        if not self.target_locked:
            self.navigator.cancelTask()
    
    def convert_to_nav_goal(self, pose):
        # Transform camera frame to map frame
        # (Requires TF2 transforms)
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        # ... transformation logic ...
        return goal

def main():
    rclpy.init()
    node = TrackerNavIntegration()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
```

---

### Example 9: Custom Gesture Commands

Extend gesture vocabulary:

```python
#!/usr/bin/env python3
"""
Example: Add custom gestures
"""

from human_tracker_ros2.gesture_detector import GestureDetector
import cv2

class CustomGestureDetector(GestureDetector):
    def detect_custom_gestures(self, hand_landmarks):
        """Add more gestures."""
        gestures = {}
        
        # Thumbs up
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        if thumb_tip.y < index_tip.y:
            gestures['thumbs_up'] = True
        
        # Peace sign (2 fingers)
        extended = self._count_extended_fingers(hand_landmarks)
        if extended == 2:
            gestures['peace'] = True
        
        # Fist
        if extended == 0:
            gestures['fist'] = True
        
        return gestures

# Usage in tracker node
# detector = CustomGestureDetector()
# gestures = detector.detect_custom_gestures(hand_landmarks)
```

---

### Example 10: Multi-Robot Coordination

```python
#!/usr/bin/env python3
"""
Example: Multiple robots track different people
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class MultiRobotCoordinator(Node):
    def __init__(self):
        super().__init__('multi_robot_coordinator')
        
        # Each robot publishes its target
        self.robot_targets = {}
        
        # Subscribe to multiple robots
        for robot_id in range(3):
            self.create_subscription(
                PoseStamped,
                f'/robot_{robot_id}/tracker/target_pose',
                lambda msg, rid=robot_id: self.target_callback(rid, msg),
                10
            )
    
    def target_callback(self, robot_id, msg):
        self.robot_targets[robot_id] = msg
        
        # Coordinate robots to avoid collisions
        self.avoid_conflicts()
    
    def avoid_conflicts(self):
        # Implement coordination logic
        # E.g., assign different zones, maintain minimum distance
        pass

def main():
    rclpy.init()
    node = MultiRobotCoordinator()
    rclpy.spin(node)
```

---

## Troubleshooting Examples

### Example 11: Debug Camera Issues

```bash
# Check camera connection
lsusb | grep Intel

# Test with RealSense viewer
realsense-viewer

# Check ROS camera topics
ros2 topic list | grep camera
ros2 topic hz /camera/color/image_raw

# View raw camera feed
ros2 run rqt_image_view rqt_image_view /camera/color/image_raw
```

---

### Example 12: Adjust Detection Sensitivity

If gestures aren't being detected:

```bash
# Lower confidence thresholds
ros2 run human_tracker_ros2 tracker_node \
    --ros-args \
    -p gesture_confidence:=0.5 \
    -p pose_confidence:=0.3
```

If getting too many false detections:

```bash
# Higher confidence thresholds
ros2 run human_tracker_ros2 tracker_node \
    --ros-args \
    -p gesture_confidence:=0.9 \
    -p pose_confidence:=0.7
```

---

### Example 13: Test Individual Components

Test gesture detection only:

```python
#!/usr/bin/env python3
from human_tracker_ros2.gesture_detector import GestureDetector
import cv2

detector = GestureDetector()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = detector.detect_gestures(frame)
    annotated = detector.draw_gesture_info(frame, result)
    
    cv2.imshow('Gesture Test', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

### Example 14: Simulate Robot Movement

Test without actual robot:

```bash
# Terminal 1: Run tracker
ros2 launch human_tracker_ros2 tracker_launch.py

# Terminal 2: Monitor commands
ros2 topic echo /cmd_vel

# Terminal 3: Visualize in RViz
rviz2
```

---

### Example 15: Performance Profiling

```python
#!/usr/bin/env python3
"""
Example: Profile tracker performance
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import time

class PerformanceProfiler(Node):
    def __init__(self):
        super().__init__('performance_profiler')
        
        self.frame_times = []
        self.last_time = None
        
        self.subscription = self.create_subscription(
            Image,
            '/tracker/annotated_image',
            self.callback,
            10
        )
    
    def callback(self, msg):
        current_time = time.time()
        
        if self.last_time is not None:
            frame_time = current_time - self.last_time
            self.frame_times.append(frame_time)
            
            if len(self.frame_times) >= 100:
                avg_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
                self.get_logger().info(f'Average FPS: {avg_fps:.1f}')
                self.frame_times = []
        
        self.last_time = current_time

def main():
    rclpy.init()
    profiler = PerformanceProfiler()
    rclpy.spin(profiler)

if __name__ == '__main__':
    main()
```

---

## Quick Reference Commands

```bash
# Launch tracker
ros2 launch human_tracker_ros2 tracker_launch.py

# View topics
ros2 topic list

# View annotated feed
ros2 run rqt_image_view rqt_image_view /tracker/annotated_image

# Monitor target pose
ros2 topic echo /tracker/target_pose

# Record session
ros2 bag record -a

# Check performance
ros2 topic hz /tracker/annotated_image

# Stop all nodes
Ctrl+C (in launch terminal)
```

---

## Tips and Best Practices

1. **Lighting**: Ensure consistent, even lighting (300-500 lux)
2. **Distance**: Optimal tracking distance is 1-4 meters
3. **Gestures**: Make deliberate, clear movements
4. **Performance**: Start with lower resolution, increase as needed
5. **Testing**: Use `test_installation.py` to verify setup
6. **Monitoring**: Watch FPS to ensure system isn't overloaded
7. **Safety**: Always test with low speed limits first

---

## Additional Resources

- [ROS 2 Documentation](https://docs.ros.org/)
- [MediaPipe Guides](https://google.github.io/mediapipe/)
- [RealSense Documentation](https://dev.intelrealsense.com/)
- [Jetson Optimization](https://developer.nvidia.com/embedded/learn/tutorials)
