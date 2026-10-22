from glob import glob

from setuptools import setup

package_name = "followme_ros"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Manuel",
    description="FollowMe person following on ROS 2",
    license="MIT",
    entry_points={"console_scripts": ["followme_node = followme_ros.followme_node:main",
            "phone_pdr_node = followme_ros.phone_pdr_node:main",
                                      "lidar_owner_fusion = followme_ros.lidar_owner_fusion:main",
                                      "steer_test = followme_ros.steer_test:main"]},
)
