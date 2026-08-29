# Project Conventions

## 1. Namespacing

All WildBridge telemetry/command topics are published under the `drone_1`
namespace. The whole stack adopts this.

`/tf` and `/tf_static` are kept abosolute as per ROS standards.

---

## 2. Package structure

Packages map 1:1 to services from deployment.mmd

```
ros2_ws/
  src/
    drone_bringup/          # top-level launch + shared config
      launch/
        bridge_camera.launch.py
        system.launch.py    # includes all service launches
      config/
        common.yaml         # shared params
      package.xml
      CMakeLists.txt

    dji_controller/         # adopted from WildBridge
      dji_controller/
        submodules/
          dji_interface.py
        controller.py
      package.xml
      setup.py

    camera_streamer/        # python pkg
      camera_streamer/
        camera_decoder_node.py
      launch/
        camera_streamer.launch.py
      config/
        camera_decoder_node.yaml
      package.xml
      setup.py

    drone_localisation/     # python pkg
      drone_localisation/
        slam_node.py
        ekf_node.py
      launch/
        localisation.launch.py
      config/
        slam/params.yaml
        ekf/params.yaml
      package.xml
      setup.py

    vslam/
      config/
        camera_calibration.yaml
        slam_node.yaml
      src/
        slam_node.cpp
      launch/
        vslam.launch.py
      CMakeLists.txt
      package.xml

    drone_navigation/       # python pkg
      drone_navigation/
        trajectory_gen_node.py
        waypoint_follower_node.py
        position_controller_node.py
      launch/
        navigation.launch.py
      config/
        trajectory_gen/params.yaml
        waypoint_follower/params.yaml
        position_controller/params.yaml
      package.xml
      setup.py

    drone_evaluation/         # dev-only
      drone_evaluation/
        comparison_node.py
      launch/
        evaluation.launch.py
      config/
        comparison/params.yaml
        domain_bridge.yaml    # ros_domains_bridge config (0 -> 12)
      package.xml
      setup.py

    drone_interfaces/        # python pkg
      msg/
        AttitudeStamped.msg
        RelativeAltitudeStamped.msg
      CMakeLists.txt
      package.xml

```

## 3. Parameters

Two mechanisms, used together:
- **Declare** every parameter in the node with a default value.
- **Override** tuned values via `config/<node>/params.yaml`, loaded by launch.

---

## 4. Layered launch structure

- **Per-package launch** (`<service>.launch.py`): brings up that service's
  nodes, applies `namespace='drone_1'`, loads that package's param files.
  Runnable in isolation for development.
- **Top-level launch** (`drone_bringup/launch/system.launch.py`): includes the
  per-package launches. Brings up the whole system.

---

## 5. Run / build commands

Build (from `ros2_ws/`):
```
colcon build --symlink-install
source install/setup.bash
```

Run one service (dev):
```
ros2 launch drone_localisation localisation.launch.py
```

Run whole system:
```
ros2 launch drone_bringup system.launch.py
```

---

## 6. Docker compose commands

- **Dev default:** `command: ["bash"]` + `stdin_open: true` + `tty: true`.
- **Eventual per-service default:** the service's launch file, e.g.
  `command: ["ros2", "launch", "drone_localisation", "localisation.launch.py"]`.
