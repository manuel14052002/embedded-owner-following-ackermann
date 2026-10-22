from setuptools import setup
import os
from glob import glob

package_name = 'a2rbot_pet_car'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Manuel',
    maintainer_email='manuel@a2rlab.dev',
    description='Jetson to ESP32 person-following robot car.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_node = a2rbot_pet_car.camera_node:main',
            'tracker_node = a2rbot_pet_car.tracker_node:main',
            'serial_bridge_node = a2rbot_pet_car.serial_bridge_node:main',
        ],
    },
)
