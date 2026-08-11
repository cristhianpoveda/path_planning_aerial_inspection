"""evaluation service — domain bridge + comparison node (dev-only)."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

NAMESPACE = "drone_1"


def generate_launch_description() -> LaunchDescription:
    bridge_config = os.path.join(
        get_package_share_directory("drone_evaluation"), "config", "domain_bridge.yaml"
    )

    domain_bridge = Node(
        package="domain_bridge",
        executable="domain_bridge",
        name="ros_domains_bridge",
        output="screen",
        arguments=[bridge_config],
    )

    # comparison_node = Node(
    #     package="drone_evaluation",
    #     executable="comparison_node",
    #     name="comparison_node",
    #     namespace=NAMESPACE,
    #     output="screen",
    # )

    return LaunchDescription([domain_bridge])#, comparison_node])
