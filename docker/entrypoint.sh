#!/usr/bin/env bash
# Invariant environment setup
set -o pipefail

source /opt/ros/humble/setup.bash || true
[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash || true

set -e
exec "$@"
