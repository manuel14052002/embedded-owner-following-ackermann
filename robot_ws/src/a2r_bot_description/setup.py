import os
from glob import glob
from setuptools import setup

package_name = 'a2r_bot_description'

setup(
    name=package_name,
    version='3.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'urdf'),   glob('urdf/*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
        (os.path.join('share', package_name, 'rviz'),   glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Davey',
    maintainer_email='davey@example.com',
    description='URDF/xacro model, RViz configs and TF publishers for A2R_bot.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'wheel_state_node = a2r_bot_description.wheel_state_node:main',
    ]},
)
