import os
from glob import glob

from setuptools import find_packages, setup

package_name = "camera_streamer"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ros",
    maintainer_email="ros@todo.todo",
    description="Skeleton package (no deps yet).",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "camera_decoder_node = camera_streamer.camera_decoder_node:main",
        ],
    },
)
