#!/usr/bin/env python3
"""
Test script to verify human tracker installation and dependencies
"""

import sys
import subprocess

def check_import(module_name, package_name=None):
    """Try to import a module and report status."""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {package_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name} - {str(e)}")
        return False

def check_command(command, description):
    """Check if a command exists."""
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        print(f"✓ {description}")
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"✗ {description}")
        return False

def main():
    print("=" * 60)
    print("Human Tracker ROS 2 - Installation Verification")
    print("=" * 60)
    print()
    
    all_good = True
    
    # Check Python version
    print("Python Version:")
    print(f"  {sys.version}")
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ required")
        all_good = False
    else:
        print("✓ Python version OK")
    print()
    
    # Check Python dependencies
    print("Python Dependencies:")
    dependencies = [
        ('cv2', 'opencv-python'),
        ('mediapipe', 'mediapipe'),
        ('pyrealsense2', 'pyrealsense2'),
        ('numpy', 'numpy'),
        ('rclpy', 'rclpy'),
    ]
    
    for module, package in dependencies:
        if not check_import(module, package):
            all_good = False
    print()
    
    # Check ROS 2
    print("ROS 2 Environment:")
    commands = [
        (['ros2', '--version'], 'ROS 2 CLI'),
        (['colcon', '--version'], 'colcon build tool'),
    ]
    
    for cmd, desc in commands:
        if not check_command(cmd, desc):
            all_good = False
    print()
    
    # Check RealSense
    print("Intel RealSense:")
    realsense_commands = [
        (['rs-enumerate-devices', '-s'], 'RealSense tools'),
    ]
    
    for cmd, desc in realsense_commands:
        if not check_command(cmd, desc):
            all_good = False
    print()
    
    # Check MediaPipe models
    print("MediaPipe Models:")
    try:
        import mediapipe as mp
        
        # Try to initialize solutions
        try:
            mp_hands = mp.solutions.hands.Hands()
            mp_hands.close()
            print("✓ MediaPipe Hands model")
        except Exception as e:
            print(f"✗ MediaPipe Hands model - {str(e)}")
            all_good = False
        
        try:
            mp_pose = mp.solutions.pose.Pose()
            mp_pose.close()
            print("✓ MediaPipe Pose model")
        except Exception as e:
            print(f"✗ MediaPipe Pose model - {str(e)}")
            all_good = False
            
    except ImportError:
        print("✗ MediaPipe not available")
        all_good = False
    print()
    
    # Check camera
    print("Camera Detection:")
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) > 0:
            for i, dev in enumerate(devices):
                print(f"✓ Found RealSense: {dev.get_info(rs.camera_info.name)}")
        else:
            print("✗ No RealSense camera detected")
            print("  Make sure camera is connected via USB 3.0")
            all_good = False
    except Exception as e:
        print(f"✗ Camera check failed - {str(e)}")
        all_good = False
    print()
    
    # Summary
    print("=" * 60)
    if all_good:
        print("✓ All checks passed! System is ready.")
        print()
        print("To start the tracker:")
        print("  ros2 launch human_tracker_ros2 tracker_launch.py")
    else:
        print("✗ Some checks failed. Please review errors above.")
        print()
        print("Installation help:")
        print("  Run: ./setup.sh")
        print("  Or see: README.md")
    print("=" * 60)
    
    return 0 if all_good else 1

if __name__ == '__main__':
    sys.exit(main())
