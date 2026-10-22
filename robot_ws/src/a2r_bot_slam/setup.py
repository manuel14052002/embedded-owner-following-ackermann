from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'a2r_bot_slam'

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
    description='Occupancy-grid SLAM with scan matching, a global costmap, and global planners A*, Dijkstra, RRT a...',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'slam_node = a2r_bot_slam.slam_node:main',
            'global_costmap_node = a2r_bot_slam.global_costmap_node:main',
            'global_planner_node = a2r_bot_slam.global_planner_node:main',
            'navigation_node = a2r_bot_slam.navigation_node:main',
            'map_saver_node = a2r_bot_slam.map_saver_node:main',
        ],
    },
)
