"""Bridge service launch — dji_node (dji_controller) + camera_decoder_node
(camera_streamer), under the drone_1 namespace.

DJI telemetry/command <-> ROS, plus the camera decoder that publishes
camera/image_raw and broadcasts the static camera tf. Runnable on its own for
development, or included by system.launch.py.
"""
from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

camera_params = os.path.join(
    get_package_share_directory("camera_streamer"), "config", "camera_decoder_node.yaml"
)

NAMESPACE = "drone_1"


def generate_launch_description() -> LaunchDescription:
    dji_node = Node(
        package="dji_controller",
        executable="dji_node",
        name="dji_node",
        namespace=NAMESPACE,
        output="screen",
    )

    camera_decoder_node = Node(
        package="camera_streamer",
        executable="camera_decoder_node",
        name="camera_decoder_node",
        namespace=NAMESPACE,
        output="screen",
        parameters=[camera_params],
    )

    return LaunchDescription([dji_node, camera_decoder_node])