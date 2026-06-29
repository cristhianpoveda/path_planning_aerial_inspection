#!/usr/bin/env bash
# Invariant environment setup; per-run command stays in compose `command`.
# Sourcing happens with `set -e` disabled because ROS setup.bash scripts are
# not -e safe (they reference optional overlay files and can return non-zero).
set -o pipefail

source /opt/ros/humble/setup.bash || true
[ -f /opt/orb_slam3_ros2/install/setup.bash ] && source /opt/orb_slam3_ros2/install/setup.bash || true
[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash || true

set -e
exec "$@"