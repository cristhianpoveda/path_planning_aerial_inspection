# Diagram 5 — Node Interface Specification

Black-box interface for each node.

Convention: parameters are DECLARED in-node with defaults; tuned values live in
`config/<node>/params.yaml`.

---

## Service: bridge

### dji_controller dji_node  (adopted from WildBridge)
Job: bidirectional bridge between the modified WildBridge and ROS over TCP/UDP.

| Direction | Topic | Type | Notes |
|-----------|-------|------|-------|
| pub | `telemetry/velocity` | `geometry_msgs/Vector3` | stamp with measurement time |
| pub | `telemetry/altitude_agl` | `std_msgs/Float64` | AGL, from KeyAltitude |
| sub | `command/stick` | `std_msgs/Float64MultiArray` | [lx, ly, rx, ry] |

Params: `phone_IP: 192.168.50.18`, `tcp_port: 8081`, `udp_port: 8082`, `http_port: 8080`,
`telemetry publish rate: 20Hz`.

### camera_streamer camera_decoder
Job: receive the encoded camera stream (single-client TCP), decode,
publish frames; broadcast the static camera extrinsic.

| Direction | Topic | Type | Notes |
|-----------|-------|------|-------|
| pub | `camera/image_raw` | `sensor_msgs/Image` | decoded frame |
| pub | `/tf_static` | `geometry_msgs/TransformStamped` | base_link->camera_link->optical (STATIC) |

Params: `camera_tcp_port: 8900`, `frame_ids: camera_link, camera_optical_frame`.

---

## Service: localisation

### localisation slam_node
Job: monocular visual odometry with scale ambiguity.

| Direction | Topic | Type | Notes |
|-----------|-------|------|-------|
| sub | `camera/image_raw` | `sensor_msgs/Image` | |
| pub | `vo/odom` | `nav_msgs/Odometry` | no metric scale |

Params: `vocabulary: <vocabulary>`, `camera calibration: <calibration>`, `feature settings: <settings>`, `devices: GPU`.

### localisation ekf_node
Job: fuse V-SLAM with metric velocity + AGL altitude to resolves scale,
outputs metric pose; broadcasts dynamic tf.

| Direction | Topic | Type | Notes |
|-----------|-------|------|-------|
| sub | `vo/odom` | `nav_msgs/Odometry` | |
| sub | `telemetry/velocity` | `geometry_msgs/Vector3` | metric velocity (scale ref) |
| sub | `telemetry/altitude_agl` | `std_msgs/Float64` | vertical metric anchor |
| pub | `localisation/pose` | `geometry_msgs/PoseWithCovarianceStamped` | source-agnostic pose |
| pub | `/tf` | `tf2_msgs/TFMessage` | map->odom->base_link |

Params: `covariances`, `scale-init: <scale-init>`, `frame ids: map, odom, base_link`.

---

## Service: navigation

### navigation trajectory_gen_node  (To be defined)
Job: turn a goal into a path/setpoint sequence. Stubbed until planner chosen.

| Direction | Topic | Type | Notes |
|-----------|-------|------|-------|
| sub | `goal` | `TBD` | goal msg type to define |
| pub | `plan/path` | `nav_msgs/Path` | |

Params: planner settings (TBD).

### navigation waypoint_follower_node
Job: walk along the path, emit the current setpoint.

| Direction | Topic | Type | Notes |
|-----------|-------|------|-------|
| sub | `plan/path` | `nav_msgs/Path` | |
| pub | `setpoint` | `geometry_msgs/PoseStamped` | current target |

Params: `waypoint reach tolerance: <tolerance>`, `lookahead: <lookahead>`.

### navigation position_controller_node
Job: closed-loop control. Given current pose + setpoint, produce stick velocity.

| Direction | Topic | Type | Notes |
|-----------|-------|------|-------|
| sub | `localisation/pose` | `geometry_msgs/PoseWithCovarianceStamped` | current pose |
| sub | `setpoint` | `geometry_msgs/PoseStamped` | target |
| pub | `command/stick` | `std_msgs/Float64MultiArray` | [lx, ly, rx, ry] |

Params: `PID gains: <gains>`, ...

Consumes tf for frame lookups.

---

## Service: evaluation  (dev-only, metrics)

### evaluation ros_domains_bridge  (domain_bridge)
Job: bridge the OptiTrack ground-truth topic from domain 0 into domain 12.

| Direction | Topic | Type | Notes |
|-----------|-------|------|-------|
| sub | `/optitrack/rigid_bodies/dji_mini4` (domain 0) | `geometry_msgs/PoseStamped` | external, absolute |
| pub | `/optitrack/rigid_bodies/dji_mini4` (domain 12) | `geometry_msgs/PoseStamped` | into custom domain |

Params: `ros domain ids: 0, 12` 0 -> Optitrack, 12 -> Stack.

### evaluation comparison_node
Job: compare localisation estimate vs ground truth, compute + log ATE/RPE.

| Direction | Topic | Type | Notes |
|-----------|-------|------|-------|
| sub | `localisation/pose` | `geometry_msgs/PoseWithCovarianceStamped` | our estimate |
| sub | `/rigid_bodies/dji_mini4` (domain 12) | `geometry_msgs/PoseStamped` | ground truth |
| pub | `eval/metrics` | `TBD` | ATE/RPE |

Params: `log path: evaluation/eval_logs`.

---

## To be defined!!
- `goal` message type — undecided.
- `eval/metrics` — publish a msg, or log-to-file only?
- EKF: full `map->odom->base_link` chain vs collapsed `map->base_link`.
- `satellite_count` — published by dji_node but no consumer until the GNSS/visual switch exists.
