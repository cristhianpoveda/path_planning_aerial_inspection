"""Bring up the DJI -> ROS bridge: H.264 decode + WildBridge dji_controller.

Replaces the old RTSP video-feed launch. One fixed phone IP, shared by both nodes.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


PHONE_IP = "192.168.50.18"   # phone IP on the lab LAN
NAMESPACE = "drone_1"       # telemetry namespace; camera stays at root


def generate_launch_description():
    pkg_share = get_package_share_directory("drone_bridge")
    params_file = os.path.join(pkg_share, "config", "bridge_params.yaml")

    # Decode node: param file sets topic/frame_id/etc; host overridden with the
    # shared IP so it can't drift from the controller's IP.
    decode_node = Node(
        package="drone_bridge",
        executable="h264_tcp_decode",
        name="h264_tcp_decode",
        output="screen",
        parameters=[params_file, {"host": PHONE_IP}],
    )

    # WildBridge telemetry/control node.
    controller_node = Node(
        package="dji_controller",
        executable="dji_node",
        namespace=NAMESPACE,
        parameters=[{"ip_rc": PHONE_IP}],
    )

    return LaunchDescription([decode_node, controller_node])