import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory('drone_navigation')
    ctrl_cfg = os.path.join(
        pkg, 'config', 'position_heading_controller', 'params.yaml')
    reg_cfg = os.path.join(pkg, 'config', 'registration', 'params.yaml')
    cam_cfg = os.path.join(
        get_package_share_directory('vslam'), 'config',
        'camera_calibration.yaml')

    use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='false')

    run_registration = DeclareLaunchArgument('run_registration',
                                             default_value='true')
    camera_calib = DeclareLaunchArgument('camera_calib', default_value=cam_cfg)
    registration_yaml = DeclareLaunchArgument('registration_yaml',
                                              default_value='registration.yaml')

    kp_xy = DeclareLaunchArgument('kp_xy', default_value='0.6')
    kp_z = DeclareLaunchArgument('kp_z', default_value='0.5')
    kp_yaw = DeclareLaunchArgument('kp_yaw', default_value='1.0')
    kd_xy = DeclareLaunchArgument('kd_xy', default_value='0.0')

    # Limits.
    v_max_xy = DeclareLaunchArgument('v_max_xy', default_value='0.5')
    v_max_z = DeclareLaunchArgument('v_max_z', default_value='0.3')
    yaw_rate_max_deg = DeclareLaunchArgument('yaw_rate_max_deg',
                                             default_value='20.0')

    yaw_source = DeclareLaunchArgument('yaw_source',
                                       default_value='dji_attitude')
    yaw_offset_deg = DeclareLaunchArgument('yaw_offset_deg',
                                           default_value='0.0')

    yaw_scale = DeclareLaunchArgument('yaw_scale', default_value='1.31')

    gate_on_degraded = DeclareLaunchArgument('gate_on_degraded',
                                             default_value='true')
    start_enabled = DeclareLaunchArgument('start_enabled',
                                          default_value='false')

    position_heading_controller_node = Node(
        package='drone_navigation',
        executable='position_heading_controller_node',
        name='position_heading_controller_node',
        namespace='drone_1',
        parameters=[ctrl_cfg, {
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'kp_xy': ParameterValue(
                LaunchConfiguration('kp_xy'), value_type=float),
            'kp_z': ParameterValue(
                LaunchConfiguration('kp_z'), value_type=float),
            'kp_yaw': ParameterValue(
                LaunchConfiguration('kp_yaw'), value_type=float),
            'kd_xy': ParameterValue(
                LaunchConfiguration('kd_xy'), value_type=float),
            'v_max_xy': ParameterValue(
                LaunchConfiguration('v_max_xy'), value_type=float),
            'v_max_z': ParameterValue(
                LaunchConfiguration('v_max_z'), value_type=float),
            'yaw_rate_max_deg': ParameterValue(
                LaunchConfiguration('yaw_rate_max_deg'), value_type=float),
            'yaw_source': LaunchConfiguration('yaw_source'),
            'yaw_offset_deg': ParameterValue(
                LaunchConfiguration('yaw_offset_deg'), value_type=float),
            'yaw_scale': ParameterValue(
                LaunchConfiguration('yaw_scale'), value_type=float),
            'gate_on_degraded': ParameterValue(
                LaunchConfiguration('gate_on_degraded'), value_type=bool),
            'start_enabled': ParameterValue(
                LaunchConfiguration('start_enabled'), value_type=bool),
        }],
        output='screen',
    )

    registration_node = Node(
        package='drone_navigation',
        executable='registration_node',
        name='registration_node',
        namespace='drone_1',
        parameters=[reg_cfg, {
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'camera_calib': LaunchConfiguration('camera_calib'),
            'output_path': LaunchConfiguration('registration_yaml'),
        }],
        condition=IfCondition(LaunchConfiguration('run_registration')),
        output='screen',
    )

    waypoint_follower_node = Node(
        package='drone_navigation',
        executable='waypoint_follower_node',
        name='waypoint_follower_node',
        namespace='drone_1',
        output='screen',
    )

    return LaunchDescription([
        use_sim_time, run_registration, camera_calib, registration_yaml,
        kp_xy, kp_z, kp_yaw, kd_xy,
        v_max_xy, v_max_z, yaw_rate_max_deg,
        yaw_source, yaw_offset_deg, yaw_scale,
        gate_on_degraded, start_enabled,
        registration_node,
        position_heading_controller_node, waypoint_follower_node,
    ])
