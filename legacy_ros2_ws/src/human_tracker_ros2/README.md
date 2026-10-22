# Human Tracker ROS 2 Package

A ROS 2 package for human tracking and following using Intel RealSense D435 camera and hand gesture recognition on Jetson Nano Orin.

## Features

- **Hand Gesture Detection**: Uses MediaPipe to detect raised hand gesture for target locking
- **Person Tracking**: Tracks locked human's pose and 3D position using MediaPipe Pose
- **Stop Gesture**: Release target by showing all 5 fingers (stop sign)
- **Autonomous Following**: Robot follows the locked target at a configurable distance
- **Live Video Streaming**: Publishes annotated video feed with tracking visualization
- **Optimized for Jetson Nano**: Configured for efficient CPU/GPU usage

## System Architecture

```
┌─────────────────┐
│  RealSense D435 │
│     Camera      │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Camera Node    │ ──> /camera/color/image_raw
│                 │ ──> /camera/depth/image_raw
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Tracker Node   │
│  ┌───────────┐  │
│  │  Gesture  │  │
│  │ Detector  │  │
│  └───────────┘  │
│  ┌───────────┐  │
│  │  Person   │  │
│  │  Tracker  │  │
│  └───────────┘  │
└────────┬────────┘
         │
         ├──> /tracker/annotated_image
         ├──> /tracker/target_pose
         ├──> /tracker/target_locked
         └──> /cmd_vel
```

## Prerequisites

### Hardware
- Jetson Nano Orin (or other Jetson platform)
- Intel RealSense D435 camera
- USB 3.0 connection for camera

### Software
- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10+

## Installation

### 1. Install ROS 2 Humble

Follow the official ROS 2 installation guide for your platform:
```bash
# Add ROS 2 repository
sudo apt update && sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

# Install ROS 2 Humble
sudo apt update
sudo apt install ros-humble-desktop
```

### 2. Install Intel RealSense SDK

```bash
# Add Intel RealSense repository
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u

# Install libraries
sudo apt install librealsense2-dkms librealsense2-utils librealsense2-dev -y
```

### 3. Install RealSense ROS 2 Wrapper (Optional - for alternative camera interface)

```bash
sudo apt install ros-humble-realsense2-camera
```

### 4. Install Python Dependencies

```bash
# Install pip packages
pip3 install --upgrade pip
pip3 install opencv-python opencv-contrib-python
pip3 install mediapipe
pip3 install pyrealsense2
pip3 install numpy
```

### 5. Install ROS 2 Package Dependencies

```bash
sudo apt install ros-humble-cv-bridge ros-humble-image-transport
```

### 6. Build the Package

```bash
# Create ROS 2 workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Clone or copy this package
cp -r /path/to/human_tracker_ros2 .

# Build
cd ~/ros2_ws
colcon build --packages-select human_tracker_ros2

# Source the workspace
source ~/ros2_ws/install/setup.bash
```

## Usage

### Quick Start

1. **Connect the RealSense camera** to your Jetson Nano via USB 3.0

2. **Test the camera** (optional):
```bash
realsense-viewer
```

3. **Launch the tracker system**:
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch human_tracker_ros2 tracker_launch.py
```

4. **Lock a target**:
   - Raise your hand in view of the camera
   - Hold it raised for 2 seconds
   - You'll see "TARGET LOCKED" on the screen
   - The robot will now follow you

5. **Release the target**:
   - Raise all 5 fingers (stop sign gesture)
   - The system will release the lock and wait for a new target

### Advanced Usage

#### Custom Parameters

Launch with custom parameters:
```bash
ros2 launch human_tracker_ros2 tracker_launch.py \
    frame_rate:=30 \
    rgb_width:=640 \
    rgb_height:=480 \
    target_follow_distance:=2.0
```

#### Run Nodes Separately

Camera node only:
```bash
ros2 run human_tracker_ros2 camera_node
```

Tracker node only (requires camera feed):
```bash
ros2 run human_tracker_ros2 tracker_node
```

#### View Published Topics

```bash
# List all topics
ros2 topic list

# View annotated video feed
ros2 run rqt_image_view rqt_image_view /tracker/annotated_image

# Monitor target pose
ros2 topic echo /tracker/target_pose

# Monitor velocity commands
ros2 topic echo /cmd_vel
```

## Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/color/image_raw` | `sensor_msgs/Image` | Raw RGB camera feed |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | Raw depth feed (16-bit) |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | RGB camera calibration |
| `/camera/depth/camera_info` | `sensor_msgs/CameraInfo` | Depth camera calibration |
| `/tracker/annotated_image` | `sensor_msgs/Image` | Video with tracking overlays |
| `/tracker/target_pose` | `geometry_msgs/PoseStamped` | 3D position of target |
| `/tracker/target_locked` | `std_msgs/Bool` | Target lock status |
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity commands for robot |

## Configuration

Edit `config/tracker_config.yaml` to customize behavior:

```yaml
tracker:
  target_follow_distance: 1.5  # Following distance in meters
  max_linear_speed: 0.5        # Maximum forward speed (m/s)
  max_angular_speed: 1.0       # Maximum rotation speed (rad/s)

gesture_detector:
  hand_raise_duration: 2.0     # Seconds to hold hand raised

person_tracker:
  model_complexity: 1          # 0=fastest, 2=most accurate
```

## Performance Optimization for Jetson Nano

### 1. Maximize CPU Usage

The code is already optimized to use available resources efficiently:
- MediaPipe uses multi-threading
- Camera runs at configurable frame rate
- Image processing is vectorized with NumPy

### 2. Enable GPU Acceleration

For better performance on Jetson, ensure CUDA is available:
```bash
# Check CUDA availability
python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
```

### 3. Reduce Resolution (if needed)

For better performance:
```bash
ros2 launch human_tracker_ros2 tracker_launch.py \
    rgb_width:=424 \
    rgb_height:=240
```

### 4. Adjust Model Complexity

Edit `config/tracker_config.yaml`:
```yaml
person_tracker:
  model_complexity: 0  # Faster but less accurate
```

### 5. Monitor Performance

```bash
# CPU/GPU usage
sudo tegrastats

# ROS 2 node statistics
ros2 topic hz /camera/color/image_raw
ros2 topic hz /tracker/annotated_image
```

## Troubleshooting

### Camera Not Detected

```bash
# Check USB connection
lsusb | grep Intel

# Test with RealSense viewer
realsense-viewer

# Check permissions
sudo usermod -a -G video $USER
# Log out and back in
```

### Low Frame Rate

- Reduce camera resolution
- Lower model complexity
- Disable depth if not needed
- Ensure USB 3.0 connection

### Person/Gesture Not Detected

- Ensure good lighting
- Check camera focus
- Adjust confidence thresholds in config
- Verify distance to camera (1-4 meters optimal)

### Robot Not Following

- Check `/cmd_vel` topic is publishing: `ros2 topic echo /cmd_vel`
- Verify target is locked: `ros2 topic echo /tracker/target_locked`
- Ensure depth data is available
- Check velocity limits in configuration

## Code Structure

```
human_tracker_ros2/
├── human_tracker_ros2/
│   ├── __init__.py
│   ├── camera_node.py          # RealSense camera interface
│   ├── tracker_node.py         # Main tracking logic
│   ├── gesture_detector.py     # Hand gesture recognition
│   └── person_tracker.py       # Person detection & tracking
├── launch/
│   └── tracker_launch.py       # Launch file
├── config/
│   └── tracker_config.yaml     # Configuration parameters
├── package.xml                 # ROS 2 package manifest
├── setup.py                    # Python setup file
└── README.md                   # This file
```

## Development

### Extending Functionality

#### Add New Gestures

Edit `gesture_detector.py` and add detection logic in `detect_gestures()` method.

#### Custom Following Behavior

Modify `compute_follow_command()` in `tracker_node.py` to implement custom control logic.

#### Multi-Target Tracking

Extend `person_tracker.py` to maintain multiple targets with priority system.

## Known Issues and Solutions

1. **MediaPipe GPU support on Jetson**: MediaPipe doesn't fully support GPU on ARM. Performance is CPU-bound.
   - Solution: Use model_complexity=0 for faster inference

2. **Depth alignment lag**: Depth and RGB alignment may introduce latency
   - Solution: Consider using RGB-only mode for faster response

3. **False positive hand raises**: Quick hand movements may trigger false detections
   - Solution: Adjust `hand_raise_duration` parameter (increase for more confidence)

## Future Improvements

- [ ] Multi-person tracking with priority system
- [ ] Custom gesture vocabulary
- [ ] Obstacle avoidance integration
- [ ] Path planning for complex environments
- [ ] RViz visualization plugin
- [ ] Dynamic gesture learning
- [ ] Integration with navigation stack
- [ ] Model quantization for faster inference

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

For issues and questions:
- Open an issue on GitHub
- Check ROS 2 documentation: https://docs.ros.org
- MediaPipe documentation: https://google.github.io/mediapipe/
- Intel RealSense documentation: https://dev.intelrealsense.com/

## Acknowledgments

- Intel RealSense SDK
- Google MediaPipe
- ROS 2 Community
- Open Robotics
