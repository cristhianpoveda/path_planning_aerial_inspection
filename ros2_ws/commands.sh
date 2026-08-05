ros2 launch wildview_bringup swarm_connection.launch.py

ros2 launch wildbridge_nav_tests navigation.launch.py

ros2 bag record -o open_loop_test_cube /drone_1/cartesian_position

ros2 topic pub /drone_1/trigger_shape std_msgs/String "{data: 'cube'}" -1

# open android studio

/opt/android-studio/bin/studio.sh

# Enable wifi debugging


ffplay -fflags nobuffer -flags low_delay -framedrop \
  -analyzeduration 0 -probesize 32 \
  -i tcp://192.168.50.18:8900

# listen raw telemetry port
nc 192.168.50.18 8081
nc 192.168.50.18 8081 \
  | jq -c '[.speed, .attitude, .altitudeAgl, .gimbalJointAttitude]' \
  | uniq -c

# gimbal joint attitude "gimbalJointAttitude":{"pitch":1.9000000000000001,"roll":0.1,"yaw":0}
# attitude "attitude":{"pitch":0,"roll":-0.8,"yaw":142.5}
