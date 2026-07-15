"""drone_navigation — per-package launch (skeleton).

Brings up this service's nodes under the drone_1 namespace. Param-file loading
(config/<node>/params.yaml) is added when the nodes gain parameters.
"""
from launch import LaunchDescription
from launch_ros.actions import Node

NAMESPACE = "drone_1"


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(package="drone_navigation", executable="trajectory_gen_node",
             namespace=NAMESPACE, name="trajectory_gen_node", output="screen"),
        Node(package="drone_navigation", executable="waypoint_follower_node",
             namespace=NAMESPACE, name="waypoint_follower_node", output="screen"),
        Node(package="drone_navigation", executable="position_controller_node",
             namespace=NAMESPACE, name="position_controller_node", output="screen"),
    ])
