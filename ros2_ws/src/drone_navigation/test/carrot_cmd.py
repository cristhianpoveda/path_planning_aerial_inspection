#!/usr/bin/env python3
"""carrot_cmd -- move the setpoint along a straight line at a commanded speed.
"""
import argparse
import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=10)


def quat_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Profile:
    """Trapezoidal distance profile along the pass direction."""

    def __init__(self, length, speed, accel):
        self.length, self.v, self.a = length, speed, accel
        self.t_ramp = speed / accel
        self.d_ramp = 0.5 * accel * self.t_ramp ** 2
        if 2.0 * self.d_ramp <= length:                  # trapezoid
            self.triangular = False
            self.d_cruise = length - 2.0 * self.d_ramp
            self.t_cruise = self.d_cruise / speed
        else:                                            # never reaches speed
            self.triangular = True
            self.v = math.sqrt(accel * length)
            self.t_ramp = self.v / accel
            self.d_ramp = 0.5 * length
            self.d_cruise = self.t_cruise = 0.0
        self.t_total = 2.0 * self.t_ramp + self.t_cruise

    def at(self, t):
        """Distance travelled and carrot speed at time t."""
        a, v, tr = self.a, self.v, self.t_ramp
        if t <= 0.0:
            return 0.0, 0.0
        if t < tr:
            return 0.5 * a * t * t, a * t
        if t < tr + self.t_cruise:
            return self.d_ramp + v * (t - tr), v
        td = t - tr - self.t_cruise
        if td < tr:
            return (self.d_ramp + self.d_cruise
                    + v * td - 0.5 * a * td * td), v - a * td
        return self.length, 0.0

    def window(self):
        """Start and end time of the constant-speed segment."""
        if self.triangular:
            return None
        return self.t_ramp, self.t_ramp + self.t_cruise

    def describe(self):
        w = self.window()
        s = (f"length {self.length:.2f} m, accel {self.a:.2f} m/s^2, "
             f"total {self.t_total:.2f} s\n")
        if self.triangular:
            s += (f"  TRIANGULAR: never reaches the commanded speed, "
                  f"peak {self.v:.2f} m/s. Lengthen the pass or lower accel.")
        else:
            s += (f"  ramp {self.d_ramp:.2f} m each end, "
                  f"cruise {self.d_cruise:.2f} m at {self.v:.2f} m/s\n"
                  f"  constant-speed window {w[0]:.2f}-{w[1]:.2f} s")
        return s


class CarrotCmd(Node):

    def __init__(self, a, profile, direction):
        super().__init__("carrot_cmd")
        self.a, self.prof, self.u = a, profile, direction
        self.start = None
        self.t0 = None
        self.done = False
        self.pub = self.create_publisher(PoseStamped, a.setpoint_topic, QOS)
        self.create_subscription(PoseWithCovarianceStamped, a.pose_topic,
                                 self.on_pose, QOS)
        self.create_timer(1.0 / a.rate, self.on_timer)
        self.get_logger().info(f"waiting for {a.pose_topic} ...")

    def on_pose(self, msg: PoseWithCovarianceStamped) -> None:
        if self.start is not None:
            return                                        # latch once
        p = msg.pose.pose.position
        yaw = quat_yaw(msg.pose.pose.orientation)
        a = self.a
        x0 = a.x0 if a.x0 is not None else p.x
        y0 = a.y0 if a.y0 is not None else p.y
        z0 = a.z0 if a.z0 is not None else p.z
        self.yaw = math.radians(a.yaw) if a.yaw is not None else yaw
        self.start = (x0, y0, z0)
        self.t0 = time.monotonic()

        d = math.dist((p.x, p.y, p.z), self.start)
        if d > a.max_jump:
            self.get_logger().error(
                f"start point is {d:.2f} m away, --max-jump is {a.max_jump}. "
                f"Fly to the start with setpoint_cmd first.")
            self.start = None
            rclpy.shutdown()
            return

        end = tuple(s + self.prof.length * ui for s, ui in
                    zip(self.start, self.u))
        self.get_logger().warning(
            f"start [{x0:+.2f} {y0:+.2f} {z0:+.2f}] -> "
            f"end [{end[0]:+.2f} {end[1]:+.2f} {end[2]:+.2f}] "
            f"yaw {math.degrees(self.yaw):+.1f} deg\n" + self.prof.describe())

    def on_timer(self) -> None:
        if self.start is None:
            return
        t = time.monotonic() - self.t0
        s, v = self.prof.at(t)
        lead = v / self.a.kp_xy
        r = s + lead
        if self.a.clamp:
            r = min(r, self.prof.length)      # carrot never leaves the pass

        if not self.done and t >= self.prof.t_total:
            self.done = True
            self.get_logger().warning("profile complete, holding")

        x, y, z = (c + r * ui for c, ui in zip(self.start, self.u))
        m = PoseStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.a.frame_id
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.pose.orientation.z = float(math.sin(self.yaw / 2.0))
        m.pose.orientation.w = float(math.cos(self.yaw / 2.0))
        self.pub.publish(m)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dx', type=float, default=0.0, help='pass vector, m')
    p.add_argument('--dy', type=float, default=0.0)
    p.add_argument('--dz', type=float, default=0.0)
    p.add_argument('--speed', type=float, required=True, help='m/s')
    p.add_argument('--accel', type=float, default=0.5, help='m/s^2')
    p.add_argument('--kp-xy', type=float, default=0.6,
                   help='controller position gain, sets the carrot lead')
    p.add_argument('--x0', type=float, default=None,
                   help='absolute start; default is the latched pose')
    p.add_argument('--y0', type=float, default=None)
    p.add_argument('--z0', type=float, default=None)
    p.add_argument('--yaw', type=float, default=None, help='deg, absolute')
    p.add_argument('--pose-topic', default='/drone_1/localisation/pose')
    p.add_argument('--setpoint-topic', default='/drone_1/setpoint')
    p.add_argument('--frame-id', default='odom')
    p.add_argument('--rate', type=float, default=20.0)
    p.add_argument('--max-speed', type=float, default=1.0)
    p.add_argument('--max-length', type=float, default=6.0)
    p.add_argument('--max-jump', type=float, default=0.5,
                   help='refuse if the start point is further than this')
    p.add_argument('--no-clamp', dest='clamp', action='store_false',
                   help='allow the carrot to run past the pass end while the '
                        'lead is applied; default is to clamp at the end')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()

    length = math.sqrt(a.dx ** 2 + a.dy ** 2 + a.dz ** 2)
    if length < 1e-6:
        sys.exit("pass vector is zero")
    if length > a.max_length:
        sys.exit(f"refusing a {length:.2f} m pass, --max-length {a.max_length}")
    if a.speed > a.max_speed:
        sys.exit(f"refusing {a.speed:.2f} m/s, --max-speed {a.max_speed}")

    u = (a.dx / length, a.dy / length, a.dz / length)
    prof = Profile(length, a.speed, a.accel)

    print(prof.describe())
    lead = a.speed / a.kp_xy
    reach = length if a.clamp else length + lead
    print(f"  lead at cruise {lead:.2f} m (kp_xy {a.kp_xy}); "
          f"carrot reaches {reach:.2f} m from the start"
          f"{' (clamped)' if a.clamp else ''}")
    if a.clamp and prof.d_cruise > 0 and lead > prof.d_ramp + prof.d_cruise:
        print("  WARNING: lead exceeds the cruise, the clamp will bite early")
    if prof.triangular:
        print("  refusing to fly a triangular profile")
        sys.exit(1)
    if a.dry_run:
        return

    rclpy.init()
    node = CarrotCmd(a, prof, u)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()