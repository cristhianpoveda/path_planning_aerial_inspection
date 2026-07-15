"""camera_streamer — per-package launch (skeleton).

Brings up this service's nodes under the drone_1 namespace. Param-file loading
(config/<node>/params.yaml) is added when the nodes gain parameters.
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
    camera_decoder_node = Node(
        package="camera_streamer",
        executable="camera_decoder_node",
        name="camera_decoder_node",
        namespace=NAMESPACE,
        output="screen",
        parameters=[camera_params],
    )

    return LaunchDescription([camera_decoder_node])
