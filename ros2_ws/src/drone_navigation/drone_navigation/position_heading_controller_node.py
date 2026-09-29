"""position_heading_controller_node -- P position + heading controller.

Closes the outer loop only. DJI's advanced virtual stick is already a velocity
controller  so this node maps a position andmheading error onto a body-frame velocity command and does nothing else.

    localisation/pose  (PoseWithCovarianceStamped, odom)  -- feedback
    setpoint           (PoseStamped, odom)                -- reference
    attitude           (AttitudeStamped)                  -- heading feedback
    localisation/status(LocalisationStatus)               -- health gate
    speed_vector       (Vector3Stamped)                   -- optional damping
    controller/enable  (Bool)                             -- arm / disarm
        ->
    command/vel        (TwistStamped, base_link)
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Bool
from geometry_msgs.msg import (PoseStamped, PoseWithCovarianceStamped,
                               TwistStamped, Vector3Stamped)
from drone_interfaces.msg import AttitudeStamped, LocalisationStatus

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=10)


def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def quat_yaw(q) -> float:
    """Yaw about z, radians, from a geometry_msgs Quaternion."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class PositionControllerNode(Node):

    def __init__(self) -> None:
        super().__init__("position_heading_controller_node")

        # ---------------------------------------------------------- params
        self.declare_parameter("rate_hz", 20.0)

        # Gains. kp_xy from controller_plant_model.md 9.
        self.declare_parameter("kp_xy", 0.4)          # 1/s
        self.declare_parameter("kp_z", 0.5)           # 1/s
        self.declare_parameter("kp_yaw", 1.0)         # 1/s
        self.declare_parameter("kd_xy", 0.0)          # damping on DJI velocity
        self.declare_parameter("ki_z", 0.0)           # leaky, z only
        self.declare_parameter("i_z_max", 0.05)       # m/s

        # Output limits.
        self.declare_parameter("v_max_xy", 0.5)       # m/s
        self.declare_parameter("v_max_z", 0.3)        # m/s
        self.declare_parameter("yaw_rate_max_deg", 20.0)
        self.declare_parameter("slew_xy", 1.0)        # m/s per s
        self.declare_parameter("slew_z", 1.0)
        self.declare_parameter("slew_yaw_deg", 60.0)

        # Deadbands: no plant dead zone above 0.01 m/s, so these exist only to stop the loop chattering against the 0.0056 m/s drift floor.
        self.declare_parameter("deadband_xy", 0.02)   # m
        self.declare_parameter("deadband_z", 0.02)    # m
        self.declare_parameter("deadband_yaw_deg", 2.0)

        # Plant compensation
        self.declare_parameter("yaw_scale", 1.31)

        # Heading source. 'dji_attitude' (default) or 'ekf_pose'. See the module docstring; the datums are not identical and the difference is logged at 0.2 Hz so it can be watched in flight.
        self.declare_parameter("yaw_source", "dji_attitude")
        self.declare_parameter("yaw_offset_deg", 0.0)

        # Health gating.
        self.declare_parameter("pose_timeout", 0.5)       # s
        self.declare_parameter("attitude_timeout", 0.5)   # s
        self.declare_parameter("setpoint_timeout", 5.0)   # s
        self.declare_parameter("gate_on_degraded", True)
        self.declare_parameter("start_enabled", False)

        g = lambda n: self.get_parameter(n).value
        self.rate_hz = float(g("rate_hz"))
        self.kp_xy = float(g("kp_xy"))
        self.kp_z = float(g("kp_z"))
        self.kp_yaw = float(g("kp_yaw"))
        self.kd_xy = float(g("kd_xy"))
        self.ki_z = float(g("ki_z"))
        self.i_z_max = float(g("i_z_max"))
        self.v_max_xy = float(g("v_max_xy"))
        self.v_max_z = float(g("v_max_z"))
        self.yaw_rate_max = math.radians(float(g("yaw_rate_max_deg")))
        self.slew_xy = float(g("slew_xy"))
        self.slew_z = float(g("slew_z"))
        self.slew_yaw = math.radians(float(g("slew_yaw_deg")))
        self.db_xy = float(g("deadband_xy"))
        self.db_z = float(g("deadband_z"))
        self.db_yaw = math.radians(float(g("deadband_yaw_deg")))
        self.yaw_scale = float(g("yaw_scale"))
        self.yaw_source = str(g("yaw_source"))
        self.yaw_offset = math.radians(float(g("yaw_offset_deg")))
        self.pose_timeout = float(g("pose_timeout"))
        self.att_timeout = float(g("attitude_timeout"))
        self.sp_timeout = float(g("setpoint_timeout"))
        self.gate_degraded = bool(g("gate_on_degraded"))
        self.enabled = bool(g("start_enabled"))

        if self.yaw_source not in ("dji_attitude", "ekf_pose"):
            self.get_logger().error(
                f"unknown yaw_source '{self.yaw_source}', using dji_attitude")
            self.yaw_source = "dji_attitude"

        # ---------------------------------------------------------- state
        self._pose = None          # (t_recv, p[3], yaw_ekf)
        self._setpoint = None      # (t_recv, p[3], yaw)
        self._att_yaw = None       # (t_recv, yaw)
        self._status = None        # (t_recv, LocalisationStatus)
        self._v_dji = None         # (t_recv, v[3]) ENU, NOT K_VEL-corrected
        self._i_z = 0.0
        self._last_cmd = np.zeros(4)   # vx, vy, vz, yaw_rate (pre-scale)
        self._last_reason = ""
        self._n_cycles = 0

        # ---------------------------------------------------- subscriptions
        self.create_subscription(PoseWithCovarianceStamped,
                                 "localisation/pose", self.on_pose, QOS)
        self.create_subscription(PoseStamped, "setpoint",
                                 self.on_setpoint, QOS)
        self.create_subscription(AttitudeStamped, "attitude",
                                 self.on_attitude, QOS)
        self.create_subscription(LocalisationStatus, "localisation/status",
                                 self.on_status, QOS)
        self.create_subscription(Vector3Stamped, "speed_vector",
                                 self.on_velocity, QOS)
        self.create_subscription(Bool, "controller/enable",
                                 self.on_enable, 10)

        self.pub_cmd = self.create_publisher(TwistStamped, "command/vel", QOS)

        self.create_timer(1.0 / self.rate_hz, self.on_timer)
        self.create_timer(5.0, self.on_diagnostics)

        self.get_logger().info(
            f"position_heading_controller up: {self.rate_hz:.0f} Hz, "
            f"kp_xy={self.kp_xy}, kp_z={self.kp_z}, kp_yaw={self.kp_yaw}, "
            f"yaw_source={self.yaw_source}, "
            f"enabled={self.enabled}")

    # ============================================================ callbacks
    def on_pose(self, msg: PoseWithCovarianceStamped) -> None:
        p = msg.pose.pose.position
        self._pose = (self._now(), np.array([p.x, p.y, p.z]),
                      quat_yaw(msg.pose.pose.orientation))

    def on_setpoint(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self._setpoint = (self._now(), np.array([p.x, p.y, p.z]),
                          quat_yaw(msg.pose.orientation))

    def on_attitude(self, msg: AttitudeStamped) -> None:
        self._att_yaw = (self._now(), -math.radians(msg.yaw))

    def on_status(self, msg: LocalisationStatus) -> None:
        self._status = (self._now(), msg)

    def on_velocity(self, msg: Vector3Stamped) -> None:
        v = msg.vector
        self._v_dji = (self._now(), np.array([v.x, v.y, v.z]))

    def on_enable(self, msg: Bool) -> None:
        if bool(msg.data) != self.enabled:
            self.get_logger().warning(
                f"controller {'ENABLED' if msg.data else 'DISABLED'}")
        self.enabled = bool(msg.data)
        if not self.enabled:
            self._i_z = 0.0

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ================================================================= loop
    def on_timer(self) -> None:
        """Runs at a fixed rate and publishes on every cycle..
        """
        self._n_cycles += 1
        cmd, reason = self._compute()
        cmd = self._slew(cmd)
        self._last_cmd = cmd
        if reason != self._last_reason:
            if reason:
                self.get_logger().warning(f"holding: {reason}")
            else:
                self.get_logger().info("tracking")
            self._last_reason = reason
        self._publish(cmd)

    def _compute(self):
        """Return (cmd[4], reason). cmd is vx, vy, vz, yaw_rate in body frame,
        yaw_rate in rad/s BEFORE the plant scale factor."""
        zero = np.zeros(4)
        now = self._now()

        if not self.enabled:
            return zero, "not enabled"
        if self._pose is None:
            return zero, "no pose"
        if now - self._pose[0] > self.pose_timeout:
            return zero, f"pose stale ({now - self._pose[0]:.2f} s)"
        if self._setpoint is None:
            return zero, "no setpoint"
        if now - self._setpoint[0] > self.sp_timeout:
            return zero, f"setpoint stale ({now - self._setpoint[0]:.1f} s)"

        yaw, why = self._heading(now)
        if yaw is None:
            return zero, why

        if self.gate_degraded:
            if self._status is None:
                return zero, "no localisation/status"
            if self._status[1].degraded:
                return zero, f"localisation degraded {list(self._status[1].flags)}"

        p = self._pose[1]
        sp = self._setpoint[1]
        e = sp - p                      # odom frame

        # ---- horizontal: rotate the odom error into body, then P + damping
        c, s = math.cos(-yaw), math.sin(-yaw)
        ex_b = c * e[0] - s * e[1]
        ey_b = s * e[0] + c * e[1]

        if math.hypot(ex_b, ey_b) < self.db_xy:
            ex_b = ey_b = 0.0

        vx = self.kp_xy * ex_b
        vy = self.kp_xy * ey_b

        if self.kd_xy > 0.0 and self._v_dji is not None:
            if now - self._v_dji[0] < 0.5:
                v = self._v_dji[1]
                vxb = c * v[0] - s * v[1]
                vyb = s * v[0] + c * v[1]
                vx -= self.kd_xy * vxb
                vy -= self.kd_xy * vyb

        n = math.hypot(vx, vy)
        if n > self.v_max_xy:
            vx *= self.v_max_xy / n
            vy *= self.v_max_xy / n

        # ---- vertical
        ez = e[2]
        if abs(ez) < self.db_z:
            ez = 0.0
        vz = self.kp_z * ez
        if self.ki_z > 0.0:
            self._i_z = float(np.clip(self._i_z + self.ki_z * ez / self.rate_hz,
                                      -self.i_z_max, self.i_z_max))
            vz += self._i_z
        vz = float(np.clip(vz, -self.v_max_z, self.v_max_z))

        # ---- heading
        e_yaw = wrap_pi(self._setpoint[2] - yaw)
        if abs(e_yaw) < self.db_yaw:
            e_yaw = 0.0
        r = float(np.clip(self.kp_yaw * e_yaw,
                          -self.yaw_rate_max, self.yaw_rate_max))

        return np.array([vx, vy, vz, r]), ""

    def _heading(self, now):
        """Current heading in the frame the position error is expressed in. 
        """
        if self.yaw_source == "dji_attitude":
            if self._att_yaw is None:
                return None, "no attitude"
            if now - self._att_yaw[0] > self.att_timeout:
                return None, f"attitude stale ({now - self._att_yaw[0]:.2f} s)"
            return wrap_pi(self._att_yaw[1] + self.yaw_offset), ""
        return wrap_pi(self._pose[2] + self.yaw_offset), ""

    def _slew(self, cmd):
        """Rate-limit the output. Bounds the step the aircraft sees when the
        loop transitions between holding and tracking."""
        dt = 1.0 / self.rate_hz
        lim = np.array([self.slew_xy, self.slew_xy,
                        self.slew_z, self.slew_yaw]) * dt
        d = np.clip(cmd - self._last_cmd, -lim, lim)
        return self._last_cmd + d

    def _publish(self, cmd) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(cmd[0])
        msg.twist.linear.y = float(cmd[1])
        msg.twist.linear.z = float(cmd[2])
        msg.twist.angular.z = float(cmd[3] * self.yaw_scale)
        self.pub_cmd.publish(msg)

    # ========================================================== diagnostics
    def on_diagnostics(self) -> None:
        if self._pose is None:
            return
        parts = [f"cycles={self._n_cycles}",
                 f"enabled={self.enabled}"]
        if self._setpoint is not None:
            e = self._setpoint[1] - self._pose[1]
            parts.append(f"err=[{e[0]:+.2f} {e[1]:+.2f} {e[2]:+.2f}] m")
            
        if self._att_yaw is not None:
            d = wrap_pi(self._pose[2] - self._att_yaw[1])
            parts.append(f"ekf_yaw-dji_yaw={math.degrees(d):+.1f} deg")
        if self._status is not None:
            st = self._status[1]
            parts.append(f"sigma_s/s={st.sigma_scale / max(st.scale, 1e-9):.3f}")
            if st.degraded:
                parts.append(f"DEGRADED {list(st.flags)}")
        self.get_logger().info(" | ".join(parts))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PositionControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
