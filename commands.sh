ros2 launch wildview_bringup swarm_connection.launch.py

ros2 launch wildbridge_nav_tests navigation.launch.py

ros2 bag record -o open_loop_test_cube /drone_1/cartesian_position

ros2 topic pub /drone_1/trigger_shape std_msgs/String "{data: 'cube'}" -1
