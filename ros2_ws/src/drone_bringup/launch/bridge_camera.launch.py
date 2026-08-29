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
        parameters=[{
            "command_mode": "vel",      # "stick" or "vel"
            "v_max_horizontal": 1.0,    # m/s
            "v_max_vertical": 0.5,      # m/s
            "yaw_rate_max": 30.0,       # deg/s
            # Resolved in the lab, see the app document section 8b.
            "sign_vx": 1.0,
            "sign_vy": 1.0,
            "sign_vz": 1.0,
            "sign_yaw": 1.0,
        }],
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