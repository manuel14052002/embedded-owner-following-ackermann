from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'a2r_bot_follow'

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
    description='Owner-following local planner. Eight interchangeable local planners (DWA, APF, TEB, VFH+, Pure Pu...',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'smart_planner_node = a2r_bot_follow.smart_planner_node:main',
            'local_planner_bridge_node = a2r_bot_follow.local_planner_bridge_node:main',
            'odometry_node = a2r_bot_follow.odometry_node:main',
            'owner_fusion_node = a2r_bot_follow.owner_fusion_node:main',
            'follow_manager_node = a2r_bot_follow.follow_manager_node:main',
            'collision_monitor_node = a2r_bot_follow.collision_monitor_node:main',
            'cmd_mux_node = a2r_bot_follow.cmd_mux_node:main',
            'scan_leg_fill_node = a2r_bot_follow.scan_leg_fill_node:main',
            'gps_owner_node = a2r_bot_follow.gps_owner_node:main',          # a2r_step3
        ],
    },
)
