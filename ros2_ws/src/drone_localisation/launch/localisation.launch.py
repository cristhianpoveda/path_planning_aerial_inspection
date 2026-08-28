import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
     pkg = get_package_share_directory('drone_localisation')
     ekf_cfg = os.path.join(pkg, 'config', 'ekf', 'params.yaml')

     inovation_log = DeclareLaunchArgument('innovation_log', default_value='')
     use_sim_true = DeclareLaunchArgument('use_sim_time', default_value='false')
     estimate_scale = DeclareLaunchArgument('estimate_scale', default_value='1.0')
     sigma_yaw = DeclareLaunchArgument('sigma_yaw', default_value='0.011')
     R_speed_h = DeclareLaunchArgument('R_speed_h', default_value='0.0011')
     q_s = DeclareLaunchArgument('q_s', default_value='1e-4')
     T_HOLD = DeclareLaunchArgument('T_HOLD', default_value='0.0')
     K_VEL = DeclareLaunchArgument('K_VEL', default_value='0.87')
     init_dump = DeclareLaunchArgument('init_dump', default_value='')

     ekf_node = Node(
          package='drone_localisation',
          executable='ekf_node',
          name='ekf_node',
          namespace='drone_1',
          parameters=[ekf_cfg, {
               'innovation_log': LaunchConfiguration('innovation_log'),
               'use_sim_time': LaunchConfiguration('use_sim_time'),
               'estimate_scale': ParameterValue(
                    LaunchConfiguration('estimate_scale'), value_type=float),
               'sigma_yaw': ParameterValue(
                    LaunchConfiguration('sigma_yaw'), value_type=float),
               'R_speed_h': ParameterValue(
                    LaunchConfiguration('R_speed_h'), value_type=float),
               'q_s': ParameterValue(
                    LaunchConfiguration('q_s'), value_type=float),
               'T_HOLD': ParameterValue(
                    LaunchConfiguration('T_HOLD'), value_type=float),
               'K_VEL': ParameterValue(
                    LaunchConfiguration('K_VEL'), value_type=float),
               'init_dump': LaunchConfiguration('init_dump'),
          }],
          output='screen',
     )

     map_odom = Node(
             package="tf2_ros",
             executable="static_transform_publisher",
             name="map_to_odom",
             arguments=[
                 "--frame-id", "map",
                 "--child-frame-id", "odom",
                 "--x", "0.0", "--y", "0.0", "--z", "0.0",
                 "--roll", "0.0",      # rad
                 "--pitch", "0.0",
                 "--yaw", "0.0",
             ],
             output="screen",
         )

     return LaunchDescription([inovation_log, use_sim_true, estimate_scale, sigma_yaw, R_speed_h, q_s, T_HOLD, K_VEL, init_dump, ekf_node, map_odom])
