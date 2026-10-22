from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'a2r_bot_pet'

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
    description='Animated robot face for PET mode. Renders eyes and mouth on the 800x480 LCD with pygame and maps ...',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'face_node = a2r_bot_pet.face_node:main',
            'state_bridge_node = a2r_bot_pet.state_bridge_node:main',
        ],
    },
)
