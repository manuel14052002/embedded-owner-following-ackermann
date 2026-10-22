from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'a2r_bot_arduino'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'docs'), glob('docs/*')),
        (os.path.join('share', package_name, 'firmware', 'a2r_bot_firmware'),
            glob('firmware/a2r_bot_firmware/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='A2R Robotics',
    maintainer_email='a2r.bot@example.com',
    description='Serial bridge between the Raspberry Pi and the Arduino Nano ESP32 firmware. Owns /dev/ttyARDUINO ...',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'serial_bridge_node = a2r_bot_arduino.serial_bridge_node:main',
            'firmware_monitor_node = a2r_bot_arduino.firmware_monitor_node:main',
            'robot_status_node = a2r_bot_arduino.robot_status_node:main',
        ],
    },
)
