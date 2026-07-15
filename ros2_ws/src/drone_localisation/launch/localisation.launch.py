"""drone_localisation — per-package launch (skeleton).

Brings up this service's nodes under the drone_1 namespace. Param-file loading
(config/<node>/params.yaml) is added when the nodes gain parameters.
"""
from launch import LaunchDescription
from launch_ros.actions import Node

NAMESPACE = "drone_1"


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(package="drone_localisation", executable="slam_node",
             namespace=NAMESPACE, name="slam_node", output="screen"),
        Node(package="drone_localisation", executable="ekf_node",
             namespace=NAMESPACE, name="ekf_node", output="screen"),
    ])
