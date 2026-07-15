"""vslam — ORB-SLAM3 monocular node."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

NAMESPACE = "drone_1"


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("vslam")
    settings = os.path.join(pkg_share, "config", "camera_calibration.yaml")
    params = os.path.join(pkg_share, "config", "slam_node.yaml")

    slam_node = Node(
        package="vslam",
        executable="slam_node",
        name="slam_node",
        namespace=NAMESPACE,
        output="screen",
        parameters=[
            params,
            {"settings_path": settings},   # absolute path, resolved at launch
        ],
    )

    return LaunchDescription([slam_node])
