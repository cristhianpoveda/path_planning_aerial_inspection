from glob import glob

from setuptools import find_packages, setup

package_name = "drone_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="cristhian",
    maintainer_email="cristhian@example.com",
    description="DJI -> ROS bridge: H.264 TCP decode and telemetry adapters.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "h264_tcp_decode = drone_bridge.h264_tcp_decode:main",
        ],
    },
)
