from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'a2r_bot_sensor_fusion'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='A2R Robotics',
    maintainer_email='a2r.bot@example.com',
    description='Kalman-filter sensor fusion: Madgwick/Mahony attitude estimation from the GY-91, an extended Kalm...',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'imu_filter_node = a2r_bot_sensor_fusion.imu_filter_node:main',
            'ekf_state_node = a2r_bot_sensor_fusion.ekf_state_node:main',
            'scan_fusion_node = a2r_bot_sensor_fusion.scan_fusion_node:main',
        ],
    },
)
