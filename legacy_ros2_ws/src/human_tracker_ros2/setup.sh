#!/bin/bash
# Setup script for Human Tracker ROS 2 Package
# Run this script on Jetson Nano Orin with Ubuntu 22.04

set -e

echo "================================================"
echo "Human Tracker ROS 2 Package - Setup Script"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Ubuntu
if [ ! -f /etc/lsb-release ]; then
    echo -e "${RED}Error: This script is designed for Ubuntu${NC}"
    exit 1
fi

source /etc/lsb-release
if [ "$DISTRIB_RELEASE" != "22.04" ]; then
    echo -e "${YELLOW}Warning: This script is tested on Ubuntu 22.04${NC}"
fi

echo -e "${GREEN}Step 1: Installing system dependencies${NC}"
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    software-properties-common

echo ""
echo -e "${GREEN}Step 2: Installing ROS 2 Humble (if not installed)${NC}"
if ! command -v ros2 &> /dev/null; then
    echo "ROS 2 not found, installing..."
    
    # Add ROS 2 repository
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y universe
    
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
    sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
    
    sudo apt update
    sudo apt install -y ros-humble-desktop
    
    # Add to bashrc
    if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
        echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
    fi
    
    source /opt/ros/humble/setup.bash
else
    echo "ROS 2 already installed"
    source /opt/ros/humble/setup.bash
fi

echo ""
echo -e "${GREEN}Step 3: Installing Intel RealSense SDK${NC}"
if ! dpkg -l | grep -q librealsense2; then
    echo "Installing RealSense SDK..."
    
    # Add repository
    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || \
    sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
    
    sudo add-apt-repository -y "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main"
    
    sudo apt update
    sudo apt install -y \
        librealsense2-dkms \
        librealsense2-utils \
        librealsense2-dev \
        librealsense2-dbg
else
    echo "RealSense SDK already installed"
fi

echo ""
echo -e "${GREEN}Step 4: Installing ROS 2 dependencies${NC}"
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-vision-msgs \
    python3-colcon-common-extensions

echo ""
echo -e "${GREEN}Step 5: Installing Python dependencies${NC}"
pip3 install --upgrade pip

# Install from requirements.txt if available, otherwise install individually
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
else
    pip3 install \
        opencv-python \
        opencv-contrib-python \
        mediapipe \
        pyrealsense2 \
        numpy \
        psutil
fi

echo ""
echo -e "${GREEN}Step 6: Setting up ROS 2 workspace${NC}"
WORKSPACE_DIR=~/ros2_ws

if [ ! -d "$WORKSPACE_DIR" ]; then
    mkdir -p $WORKSPACE_DIR/src
    echo "Created workspace at $WORKSPACE_DIR"
else
    echo "Workspace already exists at $WORKSPACE_DIR"
fi

# Copy package to workspace
PACKAGE_DIR="$WORKSPACE_DIR/src/human_tracker_ros2"
if [ ! -d "$PACKAGE_DIR" ]; then
    cp -r "$(pwd)" "$PACKAGE_DIR"
    echo "Copied package to workspace"
else
    echo "Package already in workspace, updating..."
    cp -r "$(pwd)"/* "$PACKAGE_DIR/"
fi

echo ""
echo -e "${GREEN}Step 7: Building the package${NC}"
cd $WORKSPACE_DIR
colcon build --packages-select human_tracker_ros2

echo ""
echo -e "${GREEN}Step 8: Setting up environment${NC}"
if ! grep -q "source $WORKSPACE_DIR/install/setup.bash" ~/.bashrc; then
    echo "source $WORKSPACE_DIR/install/setup.bash" >> ~/.bashrc
fi

source $WORKSPACE_DIR/install/setup.bash

echo ""
echo -e "${GREEN}Step 9: Configuring camera permissions${NC}"
# Add user to video group for camera access
sudo usermod -a -G video $USER

# Create udev rules for RealSense
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", MODE="0666"' | sudo tee /etc/udev/rules.d/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "${YELLOW}Important:${NC}"
echo "1. Log out and back in for group changes to take effect"
echo "2. Connect your RealSense D435 camera"
echo "3. Test camera with: realsense-viewer"
echo "4. Run the tracker with: ros2 launch human_tracker_ros2 tracker_launch.py"
echo ""
echo -e "${YELLOW}For Jetson Nano optimization:${NC}"
echo "  sudo nvpmodel -m 0      # Set to max performance"
echo "  sudo jetson_clocks      # Lock clocks to maximum"
echo ""
echo -e "${GREEN}Enjoy tracking!${NC}"
