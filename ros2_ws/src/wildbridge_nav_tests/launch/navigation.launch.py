from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='wildbridge_nav_tests',
            executable='velocity_odometry',
            name='velocity_odometry_node',
            output='screen'
        ),
        Node(
            package='wildbridge_nav_tests',
            executable='shape_flyer',
            name='shape_flyer_node',
            output='screen'
        )
    ])