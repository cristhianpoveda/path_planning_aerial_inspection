ros2 launch wildview_bringup swarm_connection.launch.py

ros2 launch wildbridge_nav_tests navigation.launch.py

ros2 bag record -o open_loop_test_cube /drone_1/cartesian_position

ros2 topic pub /drone_1/trigger_shape std_msgs/String "{data: 'cube'}" -1

# open android studio

/opt/android-studio/bin/studio.sh

ffplay -fflags nobuffer -flags low_delay -framedrop \
  -analyzeduration 0 -probesize 32 \
  -i tcp://192.168.50.18:8900
