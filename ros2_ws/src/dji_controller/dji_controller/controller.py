"""
Author: Edouard Rolland
Project: WildDrone
Contact: edr@mmmi.sdu.dk

This file was written as part of the WildDrone project and implements a ROS 2 node for controlling a DJI drone 
via the WildBridge app. The node handles both command reception and telemetry publishing.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String, Float64MultiArray, Float64, Int32, Bool
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Vector3Stamped, TwistStamped
from datetime import datetime
from requests.exceptions import RequestException
import socket  # Added for UDP
import json

from dji_controller.submodules.dji_interface import *
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import numpy as np
from collections import deque
from rclpy.time import Time
from drone_interfaces.msg import RelativeAltitudeStamped, AttitudeStamped

class ClockOffsetTracker:
    """Maps Wildbridge monotonic clock onto the ROS clock."""

    def __init__(self, window_ns: int = 5_000_000_000, min_samples: int = 40):
        self._window_ns = window_ns
        self._min_samples = min_samples
        self._samples = deque()          # (t_local_ns, r_ns), ordered by t_local
        self._last_phone_ns = {}
        self._last_out = {}              # per-topic monotonicity

    def is_warm(self) -> bool:
        return len(self._samples) >= self._min_samples

    def to_ros_ns(self, phone_clock_ns: int, laptop_clock_ns: int, key: str) -> int:
        # Remote clock reset -> monotonic time drops -> forget history.
        prev_phone = self._last_phone_ns.get(key)
        if prev_phone is not None and phone_clock_ns < prev_phone:
            self._samples.clear()
            self._last_out.clear()
            self._last_phone_ns.clear()
        self._last_phone_ns[key] = phone_clock_ns

        r = laptop_clock_ns - phone_clock_ns
        self._samples.append((laptop_clock_ns, r))

        cutoff = laptop_clock_ns - self._window_ns
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        offset_ns = min(r for _, r in self._samples)   # delay floor ~= O
        out_ns = phone_clock_ns + offset_ns

        # The offset steps up when the window minimum ages out; never let a
        # later message on the same topic carry an earlier stamp.
        prev = self._last_out.get(key)
        if prev is not None and out_ns <= prev:
            out_ns = prev + 1
        self._last_out[key] = out_ns
        return out_ns

class DjiNode(Node):
    def __init__(self):
        super().__init__('DjiNode')
        self.get_logger().info("Node Initialisation")

        # Retrieve the drone's IP address from the parameter server
        self.declare_parameter('ip_rc', '192.168.50.18')  # Default IP
        self.ip_rc = self.get_parameter(
            'ip_rc').get_parameter_value().string_value

        # Initialize the DJI drone interface
        self.dji_interface = DJIInterface(self.ip_rc)

        # Verify the connection to the drone
        self.connected = self.verify_connection()
        if not self.connected:
            self.get_logger().error(
                f"Unable to connect to the drone at IP: {self.ip_rc}. Shutting down node.")
            self.get_logger().info("Connection Failure")
            return
        
        # UDP initialisation
        self.udp_port = 8082
        self.udp_seq_num = 0
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.get_logger().info(f"UDP Socket ready to target {self.ip_rc}:{self.udp_port}")

        # --- Command mode and velocity limits ---
        # 'stick'  : [-1, 1] normalised RC deflection  (DJI normal virtual stick)
        # 'vel'    : body-frame m/s and deg/s          (DJI advanced virtual stick)
        self.declare_parameter('command_mode', 'stick')
        self.command_mode = self.get_parameter(
            'command_mode').get_parameter_value().string_value

        if self.command_mode not in ('stick', 'vel'):
            self.get_logger().error(
                f"Invalid command_mode '{self.command_mode}', falling back to 'stick'")
            self.command_mode = 'stick'

        # Operating limits for the 5 x 4 x 3 m arena. The app clamps again.
        self.declare_parameter('v_max_horizontal', 1.0)     # m/s
        self.declare_parameter('v_max_vertical', 0.5)       # m/s
        self.declare_parameter('yaw_rate_max', 30.0)        # deg/s
        self.v_max_horizontal = self.get_parameter(
            'v_max_horizontal').get_parameter_value().double_value
        self.v_max_vertical = self.get_parameter(
            'v_max_vertical').get_parameter_value().double_value
        self.yaw_rate_max = self.get_parameter(
            'yaw_rate_max').get_parameter_value().double_value

        # Axis signs, resolved in the lab. See the app document, section 8b.
        self.declare_parameter('sign_vx', 1.0)
        self.declare_parameter('sign_vy', 1.0)
        self.declare_parameter('sign_vz', 1.0)
        self.declare_parameter('sign_yaw', 1.0)
        self.sign_vx = self.get_parameter('sign_vx').get_parameter_value().double_value
        self.sign_vy = self.get_parameter('sign_vy').get_parameter_value().double_value
        self.sign_vz = self.get_parameter('sign_vz').get_parameter_value().double_value
        self.sign_yaw = self.get_parameter('sign_yaw').get_parameter_value().double_value

        self.get_logger().info(
            f"Command mode: {self.command_mode} | "
            f"limits: {self.v_max_horizontal} m/s h, {self.v_max_vertical} m/s v, "
            f"{self.yaw_rate_max} deg/s yaw")

        # Start the telemetry stream (TCP socket on port 8081)
        self.dji_interface.startTelemetryStream()

        # Subscribers for drone commands with Empty messages
        self.create_subscription(
            Empty, 'command/takeoff', self.takeoff_callback, 10)
        self.create_subscription(Empty, 'command/land', self.land_callback, 10)
        self.create_subscription(Empty, 'command/rth', self.rth_callback, 10)
        self.create_subscription(
            Empty, 'command/abort_mission', self.abort_mission_callback, 10)
        self.create_subscription(
            Empty, 'command/enable_virtual_stick', self.enable_virtual_stick_callback, 10)
        self.create_subscription(
            Empty, 'command/abort_dji_native_mission', self.abort_dji_native_mission_callback, 10)

        # Subscribers for drone commands with specific messages
        self.create_subscription(
            Float64MultiArray, 'command/goto_waypoint', self.goto_waypoint_callback, 10)
        self.create_subscription(
            Float64MultiArray, 'command/goto_waypoint_pid_tuning', self.goto_waypoint_pid_tuning_callback, 10)

        self.create_subscription(
            String, 'command/goto_trajectory', self.goto_trajectory_callback, 10)
        self.create_subscription(
            String, 'command/goto_trajectory_dji_native', self.goto_trajectory_dji_native_callback, 10)

        self.create_subscription(
            Float64, 'command/goto_yaw', self.goto_yaw_callback, 10)
        self.create_subscription(
            Float64, 'command/goto_altitude', self.goto_altitude_callback, 10)
        self.create_subscription(
            Float64, 'command/gimbal_pitch', self.gimbal_pitch_callback, 10)
        self.create_subscription(
            Float64, 'command/gimbal_yaw', self.gimbal_yaw_callback, 10)
        self.create_subscription(
            Float64, 'command/zoom_ratio', self.zoom_ratio_callback, 10)
        self.create_subscription(
            Float64, 'command/set_rth_altitude', self.set_rth_altitude_callback, 10)
        
        # Control subscriber. Exactly one of the two is created, so a stale
        # publisher on the other topic can never reach the aircraft.
        if self.command_mode == 'vel':
            # Body frame. linear.x forward, linear.y lateral, linear.z up,
            # angular.z yaw rate in rad/s (converted to deg/s before sending).
            self.create_subscription(
                TwistStamped, 'command/vel', self.vel_callback, 10)
            self.get_logger().info("Subscribed to command/vel (advanced velocity mode)")
        else:
            # Virtual stick control subscriber (leftX, leftY, rightX, rightY)
            self.create_subscription(
                Float64MultiArray, 'command/stick', self.stick_callback, 10)
            self.get_logger().info("Subscribed to command/stick (normal stick mode)")

        # Subscribers for camera commands
        self.create_subscription(
            Empty, 'command/camera/start_recording', self.start_recording_callback, 10)
        self.create_subscription(
            Empty, 'command/camera/stop_recording', self.stop_recording_callback, 10)
        
        self._clock_offset = ClockOffsetTracker()   # shared: phone monotonic clock
        self._last_pkt_tphone = None       # flight-controller packet (~10 Hz)
        self._last_gimbal_tphone = None    # gimbal packet (~17.5 Hz)
        self._last_rx_time = None
        self._pkt_period_ns = int(1e9 / 10.0)
        self._gimbal_period_ns = int(1e9 / 17.5)

        # Publishers for telemetry
        self.speed_pub = self.create_publisher(Float64, 'speed', 10)
        self.speed_vector_pub = self.create_publisher(Vector3Stamped, 'speed_vector', 10)
        self.heading_pub = self.create_publisher(Float64, 'heading', 10)
        self.attitude_pub = self.create_publisher(AttitudeStamped, 'attitude', 10)
        self.location_pub = self.create_publisher(NavSatFix, 'location', 10)
        self.gimbal_attitude_pub = self.create_publisher(
            String, 'gimbal_attitude', 10)
        self.gimbal_joint_attitude_pub = self.create_publisher(
            AttitudeStamped, 'gimbal_joint_attitude', 10)
        self.zoom_fl_pub = self.create_publisher(Float64, 'zoom_fl', 10)
        self.hybrid_fl_pub = self.create_publisher(Float64, 'hybrid_fl', 10)
        self.optical_fl_pub = self.create_publisher(Float64, 'optical_fl', 10)
        self.zoom_ratio_pub = self.create_publisher(Float64, 'zoom_ratio', 10)
        self.battery_level_pub = self.create_publisher(
            Float64, 'battery_level', 10)
        self.satellite_count_pub = self.create_publisher(
            Int32, 'satellite_count', 10)

        self.gimbal_yaw_pub = self.create_publisher(Float64, 'gimbal_yaw', 10)
        self.gimbal_pitch_pub = self.create_publisher(
            Float64, 'gimbal_pitch', 10)

        # Mission status publishers
        self.waypoint_reached_pub = self.create_publisher(
            Bool, 'waypoint_reached', 10)
        self.intermediary_waypoint_reached_pub = self.create_publisher(
            Bool, 'intermediary_waypoint_reached', 10)
        self.altitude_reached_pub = self.create_publisher(
            Bool, 'altitude_reached', 10)
        self.yaw_reached_pub = self.create_publisher(
            Bool, 'yaw_reached', 10)

        # Home location publishers
        self.home_location_pub = self.create_publisher(
            NavSatFix, 'home_location', 10)
        self.home_set_pub = self.create_publisher(
            Bool, 'home_set', 10)
        self.distance_to_home_pub = self.create_publisher(
            Float64, 'distance_to_home', 10)

        # Flight time publishers
        self.remaining_flight_time_pub = self.create_publisher(
            Float64, 'remaining_flight_time', 10)
        self.time_needed_to_go_home_pub = self.create_publisher(
            Float64, 'time_needed_to_go_home', 10)
        self.time_needed_to_land_pub = self.create_publisher(
            Float64, 'time_needed_to_land', 10)
        self.time_to_landing_spot_pub = self.create_publisher(
            Float64, 'time_to_landing_spot', 10)
        self.max_radius_can_fly_and_go_home_pub = self.create_publisher(
            Float64, 'max_radius_can_fly_and_go_home', 10)
        
        # Battery needed publishers
        self.battery_needed_to_go_home_pub = self.create_publisher(
            Float64, 'battery_needed_to_go_home', 10)
        self.battery_needed_to_land_pub = self.create_publisher(
            Float64, 'battery_needed_to_land', 10)

        # Camera Publisher
        self.camera_is_recording_pub = self.create_publisher(
            Bool, 'camera/is_recording', 10)
        
        # Relative altitude
        self.relative_altitude_pub = self.create_publisher(
            RelativeAltitudeStamped, 'relative_altitude', 10)

        # Timer to publish telemetry at regular intervals
        # Publish every 1/20 second (50ms)
        self.create_timer(0.05, self.publish_states)

        self.get_logger().info(
            f"DroneNode initialized and connected to IP: {self.ip_rc}")

    def _ros_stamp(self, stamp_phone_ns: int, laptop_read_time_ns: int, key: str):
        ros_ns = self._clock_offset.to_ros_ns(
            int(stamp_phone_ns), int(laptop_read_time_ns), key)
        return Time(nanoseconds=ros_ns).to_msg()

    ##############################
    # Connection Verification    #
    ##############################

    def verify_connection(self):
        """Verify the connection to the drone by sending a test request."""
        timeout_duration = 5  # Timeout in seconds

        def connection_attempt():
            try:
                # Try to send a simple request to verify connection
                response = self.dji_interface.requestSend("/", "", verbose=True)
                self.get_logger().info(f"Connection attempt response: {response}")
                return True
            except RequestException as e:
                self.get_logger().error(f"Connection failed: {e}")
                return False
            except Exception as e:
                self.get_logger().error(f"Connection failed with unexpected error: {e}")
                return False

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(connection_attempt)
            try:
                return future.result(timeout=timeout_duration)
            except TimeoutError:
                self.get_logger().error(
                    f"Connection to {self.ip_rc} timed out after {timeout_duration} seconds.")
                return False

    ################################
    # Callbacks for drone commands #
    ################################

    def takeoff_callback(self, msg):
        self.get_logger().info("Received takeoff command.")
        self.dji_interface.requestSendTakeOff()

    def land_callback(self, msg):
        self.get_logger().info("Received land command.")
        self.dji_interface.requestSendLand()

    def rth_callback(self, msg):
        self.get_logger().info("Received return to home command.")
        self.dji_interface.requestSendRTH()

    def abort_mission_callback(self, msg):
        self.get_logger().info("Received abort mission command.")
        self.dji_interface.requestAbortMission()

    def enable_virtual_stick_callback(self, msg):
        self.get_logger().info("Received enable virtual stick command.")
        self.dji_interface.requestSendEnableVirtualStick()

    def abort_dji_native_mission_callback(self, msg):
        self.get_logger().info("Received abort DJI native mission command.")
        self.dji_interface.requestAbortDJINativeMission()

    def goto_waypoint_callback(self, msg: Float64MultiArray):
        """Navigate to waypoint with PID control.
        Expected: [lat, lon, alt, yaw] or [lat, lon, alt, yaw, speed]
        """
        self.get_logger().info("Received goto waypoint command.")
        data = msg.data
        if len(data) >= 4:
            latitude, longitude, altitude, yaw = data[:4]
            speed = data[4] if len(data) >= 5 else 5.0  # Default 5 m/s
            self.get_logger().info(
                f'Received: lat={latitude}, lon={longitude}, alt={altitude}, yaw={yaw}, speed={speed}')
        else:
            self.get_logger().warning('Received an array with fewer than 4 elements.')
            return

        self.dji_interface.requestSendGoToWPwithPID(
            latitude, longitude, altitude, yaw, speed)

    def goto_waypoint_pid_tuning_callback(self, msg: Float64MultiArray):
        """Navigate to waypoint with custom PID tuning parameters.
        Expected: [lat, lon, alt, yaw, kp_pos, ki_pos, kd_pos, kp_yaw, ki_yaw, kd_yaw]
        """
        self.get_logger().info("Received goto waypoint with PID tuning command.")
        data = msg.data
        if len(data) >= 10:
            lat, lon, alt, yaw, kp_pos, ki_pos, kd_pos, kp_yaw, ki_yaw, kd_yaw = data[:10]
            self.get_logger().info(
                f'Waypoint: ({lat}, {lon}, {alt}), Yaw: {yaw}, PID_pos: ({kp_pos}, {ki_pos}, {kd_pos}), PID_yaw: ({kp_yaw}, {ki_yaw}, {kd_yaw})')
            self.dji_interface.requestSendGoToWPwithPIDtuning(
                lat, lon, alt, yaw, kp_pos, ki_pos, kd_pos, kp_yaw, ki_yaw, kd_yaw)
        else:
            self.get_logger().warning('Received an array with fewer than 10 elements for PID tuning.')

    def goto_trajectory_callback(self, msg: String):
        """Navigate through a trajectory.
        Expected format: list of (lat, lon, alt) tuples with optional final yaw.
        Example: "[(lat1,lon1,alt1), (lat2,lon2,alt2), ...], finalYaw" or
                 "[(lat1,lon1,alt1), (lat2,lon2,alt2), ...]"
        """
        self.get_logger().info("Received goto trajectory command.")
        data = ast.literal_eval(msg.data)
        
        # Handle both formats: just waypoints list, or (waypoints, finalYaw) tuple
        if isinstance(data, tuple) and len(data) == 2:
            waypoints, finalYaw = data
        else:
            waypoints = data
            finalYaw = 0.0  # Default yaw if not provided
        
        self.get_logger().info(f"Received waypoints: {waypoints}, finalYaw: {finalYaw}")
        self.dji_interface.requestSendNavigateTrajectory(waypoints, finalYaw)

    def goto_trajectory_dji_native_callback(self, msg: String):
        """Navigate using DJI's native waypoint mission system.
        Expected format: "(speed, [(lat, lon, alt), (lat, lon, alt), ...])"
        or legacy format: "[(lat, lon, alt), (lat, lon, alt), ...]"
        """
        self.get_logger().info("Received DJI native trajectory command.")
        data = ast.literal_eval(msg.data)
        
        # Support both formats: (speed, waypoints) tuple or just waypoints list
        if isinstance(data, tuple) and len(data) == 2:
            speed, waypoints = data
        else:
            # Legacy format: just waypoints, use default speed
            waypoints = data
            speed = 10.0
        
        self.get_logger().info(f"Received DJI native waypoints: {waypoints}, speed: {speed} m/s")
        self.dji_interface.requestSendNavigateTrajectoryDJINative(waypoints, speed)

    def goto_yaw_callback(self, msg):
        self.get_logger().info("Received goto yaw command.")
        self.dji_interface.requestSendGotoYaw(msg.data)

    def goto_altitude_callback(self, msg):
        self.get_logger().info("Received goto altitude command.")
        self.dji_interface.requestSendGotoAltitude(msg.data)

    def gimbal_pitch_callback(self, msg):
        self.get_logger().info("Received gimbal pitch command.")
        self.dji_interface.requestSendGimbalPitch(msg.data)

    def gimbal_yaw_callback(self, msg):
        self.get_logger().info("Received gimbal yaw command.")
        self.dji_interface.requestSendGimbalYaw(msg.data)

    def zoom_ratio_callback(self, msg):
        self.get_logger().info("Received zoom ratio command.")
        self.dji_interface.requestSendZoomRatio(msg.data)

    def set_rth_altitude_callback(self, msg):
        self.get_logger().info("Received set RTH altitude command.")
        self.dji_interface.requestSetRTHAltitude(msg.data)

    # def stick_callback(self, msg: Float64MultiArray):
    #     """Virtual stick control. Expected: [leftX, leftY, rightX, rightY] in range [-1, 1]."""
    #     data = msg.data
    #     if len(data) >= 4:
    #         leftX, leftY, rightX, rightY = data[:4]
    #         self.dji_interface.requestSendStick(leftX, leftY, rightX, rightY)
    #     else:
    #         self.get_logger().warning('Stick command requires 4 values: leftX, leftY, rightX, rightY')

    def _send_udp(self, payload: dict, label: str):
        """Increment the sequence number and fire one datagram at the app.

        Fire and forget. The app's 200 ms watchdog is what protects the
        aircraft if this stops, so there is deliberately no retry and no
        repeater timer here.
        """
        self.udp_seq_num += 1
        payload["seq"] = self.udp_seq_num
        try:
            self.udp_sock.sendto(
                json.dumps(payload).encode('utf-8'),
                (self.ip_rc, self.udp_port))
        except Exception as e:
            self.get_logger().error(f"Failed to send UDP {label} command: {e}")

    def stick_callback(self, msg: Float64MultiArray):
        """Virtual stick control. Expected: [leftX, leftY, rightX, rightY] in range [-1, 1]."""
        data = msg.data
        if len(data) < 4:
            self.get_logger().warning(
                'Stick command requires 4 values: leftX, leftY, rightX, rightY')
            return

        leftX, leftY, rightX, rightY = data[:4]
        if not all(np.isfinite(v) for v in (leftX, leftY, rightX, rightY)):
            self.get_logger().error("Non-finite stick command dropped")
            return

        self._send_udp({
            "mode": "stick",
            "lx": float(np.clip(leftX, -1.0, 1.0)),
            "ly": float(np.clip(leftY, -1.0, 1.0)),
            "rx": float(np.clip(rightX, -1.0, 1.0)),
            "ry": float(np.clip(rightY, -1.0, 1.0)),
        }, "stick")

    def vel_callback(self, msg: TwistStamped):
        """Body-frame velocity command for the DJI advanced virtual stick.

        twist.linear.x  forward, m/s     twist.linear.y  lateral, m/s
        twist.linear.z  up, m/s          twist.angular.z yaw rate, rad/s

        The header is deliberately ignored here. It exists for the evaluation
        bags, where the controller's own stamp separates control-loop jitter
        from transport jitter. A staleness check on it would create a second
        dead-man switch competing with the app's 200 ms watchdog.

        Sent to the app in m/s and deg/s. Axis signs are parameters because the
        DJI conventions are resolved in flight, not from documentation.
        """
        t = msg.twist
        vx, vy, vz, wz = (t.linear.x, t.linear.y, t.linear.z, t.angular.z)

        if not all(np.isfinite(v) for v in (vx, vy, vz, wz)):
            self.get_logger().error("Non-finite velocity command dropped")
            return

        yaw_deg = np.degrees(wz)

        self._send_udp({
            "mode": "vel",
            "vx": float(np.clip(self.sign_vx * vx,
                                -self.v_max_horizontal, self.v_max_horizontal)),
            "vy": float(np.clip(self.sign_vy * vy,
                                -self.v_max_horizontal, self.v_max_horizontal)),
            "vz": float(np.clip(self.sign_vz * vz,
                                -self.v_max_vertical, self.v_max_vertical)),
            "yr": float(np.clip(self.sign_yaw * yaw_deg,
                                -self.yaw_rate_max, self.yaw_rate_max)),
        }, "velocity")

    def start_recording_callback(self, msg):
        self.get_logger().info("Received start recording command.")
        response = self.dji_interface.requestCameraStartRecording()
        if response:
            self.get_logger().info("Camera recording started successfully.")
        else:
            self.get_logger().error("Failed to start camera recording.")

    def stop_recording_callback(self, msg):
        self.get_logger().info("Received stop recording command.")
        response = self.dji_interface.requestCameraStopRecording()
        if response:
            self.get_logger().info("Camera recording stopped successfully.")
        else:
            self.get_logger().error("Failed to stop camera recording.")

    ##############################
    # Telemetry Publishers       #
    ##############################

    def publish_states(self):
        try:
            # Get telemetry from TCP socket stream
            telemetry = self.dji_interface.getTelemetry()
            read_time = self.get_clock().now().nanoseconds   # AFTER the read

            if not telemetry:
                return  # No telemetry data available yet

            # Transport health: packets arriving at all, regardless of content.
            if self._last_rx_time is not None:
                rx_gap = read_time - self._last_rx_time
                if rx_gap > 500_000_000:      # 500 ms
                    self.get_logger().warn(
                        f"telemetry transport gap: {rx_gap / 1e6:.0f} ms")
            self._last_rx_time = read_time

            stamps_valid = self._clock_offset.is_warm()

            # --- flight-controller packet: speed, attitude, relativeAltitude ---
            pkt_tphone = telemetry.get('pktTMonoNs', 0)
            pkt_fresh = bool(pkt_tphone) and pkt_tphone != self._last_pkt_tphone
            pkt_stamp = None
            if pkt_fresh:
                if self._last_pkt_tphone is not None:
                    gap = pkt_tphone - self._last_pkt_tphone
                    if gap > 2.5 * self._pkt_period_ns:
                        self.get_logger().warn(f"FC telemetry gap: {gap / 1e6:.0f} ms")
                self._last_pkt_tphone = pkt_tphone
                pkt_stamp = self._ros_stamp(pkt_tphone, read_time, 'pkt')

            # --- gimbal packet ---
            gim_tphone = telemetry.get('gimbalTMonoNs', 0)
            gim_fresh = bool(gim_tphone) and gim_tphone != self._last_gimbal_tphone
            gim_stamp = None
            if gim_fresh:
                if self._last_gimbal_tphone is not None:
                    gap = gim_tphone - self._last_gimbal_tphone
                    if gap > 2.5 * self._gimbal_period_ns:
                        self.get_logger().warn(f"gimbal telemetry gap: {gap / 1e6:.0f} ms")
                self._last_gimbal_tphone = gim_tphone
                gim_stamp = self._ros_stamp(gim_tphone, read_time, 'gimbal')
            
            # Speed (scalar and vector)
            if pkt_fresh and stamps_valid:
                speed_data = telemetry.get('speed', {})
                msg = Vector3Stamped()
                msg.header.stamp = pkt_stamp
                msg.header.frame_id = 'dji_ned'
                msg.vector.x = float(speed_data.get('x', 0.0))
                msg.vector.y = float(speed_data.get('y', 0.0))
                msg.vector.z = float(speed_data.get('z', 0.0))
                self.speed_vector_pub.publish(msg)

                speed = np.sqrt(msg.vector.x**2 + msg.vector.y**2 + msg.vector.z**2)
                self.speed_pub.publish(Float64(data=speed))
            
            # Heading
            self.heading_pub.publish(Float64(data=float(telemetry.get('heading', 0.0))))
            
            # Attitude
            if pkt_fresh and stamps_valid:
                att = telemetry.get('attitude', {})
                msg = AttitudeStamped()
                msg.header.stamp = pkt_stamp
                msg.roll = float(att.get('roll', 0.0))
                msg.pitch = float(att.get('pitch', 0.0))
                msg.yaw = float(att.get('yaw', 0.0))
                self.attitude_pub.publish(msg)
            
            # Location
            location = telemetry.get('location', {})
            self.location_pub.publish(NavSatFix(
                latitude=float(location.get('latitude', 0.0)),
                longitude=float(location.get('longitude', 0.0)),
                altitude=float(location.get('altitude', 0.0))
            ))
            
            # Gimbal
            gimbal_attitude = telemetry.get('gimbalAttitude', {})
            self.gimbal_attitude_pub.publish(String(data=str(gimbal_attitude)))
            if gim_fresh and stamps_valid:
                gj = telemetry.get('gimbalJointAttitude', {})
                msg = AttitudeStamped()
                msg.header.stamp = gim_stamp
                msg.roll = float(gj.get('roll', 0.0))
                msg.pitch = float(gj.get('pitch', 0.0))
                msg.yaw = float(gj.get('yaw', 0.0))
                self.gimbal_joint_attitude_pub.publish(msg)
            self.gimbal_yaw_pub.publish(
                Float64(data=float(gimbal_attitude.get('yaw', 0.0))))
            self.gimbal_pitch_pub.publish(
                Float64(data=float(gimbal_attitude.get('pitch', 0.0))))
            
            # Camera zoom
            self.zoom_fl_pub.publish(Float64(data=float(telemetry.get('zoomFl', -1))))
            self.hybrid_fl_pub.publish(Float64(data=float(telemetry.get('hybridFl', -1))))
            self.optical_fl_pub.publish(Float64(data=float(telemetry.get('opticalFl', -1))))
            self.zoom_ratio_pub.publish(Float64(data=float(telemetry.get('zoomRatio', 1.0))))
            
            # Battery and satellites
            self.battery_level_pub.publish(
                Float64(data=float(telemetry.get('batteryLevel', -1))))
            self.satellite_count_pub.publish(
                Int32(data=int(telemetry.get('satelliteCount', -1))))
            
            # Mission status (using new telemetry-based methods)
            self.waypoint_reached_pub.publish(
                Bool(data=telemetry.get('waypointReached', False)))
            self.intermediary_waypoint_reached_pub.publish(
                Bool(data=telemetry.get('intermediaryWaypointReached', False)))
            self.altitude_reached_pub.publish(
                Bool(data=telemetry.get('altitudeReached', False)))
            self.yaw_reached_pub.publish(
                Bool(data=telemetry.get('yawReached', False)))
            
            # Home location
            home_location = telemetry.get('homeLocation', {})
            self.home_location_pub.publish(NavSatFix(
                latitude=float(home_location.get('latitude', 0.0)),
                longitude=float(home_location.get('longitude', 0.0)),
                altitude=0.0  # Home location typically doesn't include altitude
            ))
            self.home_set_pub.publish(Bool(data=telemetry.get('homeSet', False)))
            self.distance_to_home_pub.publish(
                Float64(data=float(telemetry.get('distanceToHome', 0.0))))
            
            # Flight time information
            self.remaining_flight_time_pub.publish(
                Float64(data=float(telemetry.get('remainingFlightTime', 0))))
            self.time_needed_to_go_home_pub.publish(
                Float64(data=float(telemetry.get('timeNeededToGoHome', 0))))
            self.time_needed_to_land_pub.publish(
                Float64(data=float(telemetry.get('timeNeededToLand', 0))))
            self.time_to_landing_spot_pub.publish(
                Float64(data=float(telemetry.get('totalTime', 0))))
            self.max_radius_can_fly_and_go_home_pub.publish(
                Float64(data=float(telemetry.get('maxRadiusCanFlyAndGoHome', 0))))
            
            # Battery needed information
            self.battery_needed_to_go_home_pub.publish(
                Float64(data=float(telemetry.get('batteryNeededToGoHome', 0))))
            self.battery_needed_to_land_pub.publish(
                Float64(data=float(telemetry.get('batteryNeededToLand', 0))))
            
            # Camera recording status
            self.camera_is_recording_pub.publish(
                Bool(data=telemetry.get('isRecording', False)))
            
            # Relative altitude
            if pkt_fresh and stamps_valid:
                alt_val = telemetry.get('relativeAltitude')
                if alt_val is not None:
                    msg = RelativeAltitudeStamped()
                    msg.header.stamp = pkt_stamp
                    msg.altitude = float(alt_val)
                    self.relative_altitude_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error while publishing states: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DjiNode()
    try:
        if node.connected:
            rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
