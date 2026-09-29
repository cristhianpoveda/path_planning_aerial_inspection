import os
from glob import glob

from setuptools import find_packages, setup

package_name = "drone_navigation"


def config_data_files():
    """Install config/<node>/*.yaml preserving the directory structure.

    conventions.md 3 puts tuned overrides in config/<node>/params.yaml, and
    navigation.launch.py builds that path with get_package_share_directory,
    so the subdirectory has to survive the install. A flat glob would collapse
    every node's params.yaml onto the same destination.
    """
    out = []
    for root, _dirs, files in os.walk("config"):
        yamls = [os.path.join(root, f) for f in files
                 if f.endswith((".yaml", ".yml"))]
        if yamls:
            out.append((os.path.join("share", package_name, root), yamls))
    return out


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ] + config_data_files(),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ros",
    maintainer_email="ros@todo.todo",
    description="Navigation service: registration, position and heading control, waypoint following.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "waypoint_follower_node = drone_navigation.waypoint_follower_node:main",
            "position_heading_controller_node = drone_navigation.position_heading_controller_node:main",
            "registration_node = drone_navigation.registration_node:main",
            # the planner is ROS-free; this entry point is a convenience only
            "plan_tour = drone_navigation.planning.__main__:main",
        ],
    },
)
