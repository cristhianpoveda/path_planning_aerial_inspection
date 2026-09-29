#!/usr/bin/env python3
"""panel_cmd -- position and fly passes in the PANEL frame.

Everything is expressed relative to the panel rigid body, so nothing depends on
how the board happens to be oriented in the arena.

    panel frame:  --normal-axis / --flip-normal give the outward normal n,
                  the aircraft always sits on +n
                  tangent  t = normalise(z_world x n)      horizontal
                  target   = panel_origin + d*n + s*t + dz*z_world
                  yaw      = atan2(-n_y, -n_x)             faces the panel
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


def quat_to_R(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def unit(v):
    n = math.sqrt(sum(c * c for c in v))
    if n < 1e-9:
        raise ValueError("degenerate vector")
    return [c / n for c in v]


class Profile:
    """Trapezoidal distance profile along the pass direction."""

    def __init__(self, length, speed, accel):
        self.length, self.v, self.a = length, speed, accel
        self.t_ramp = speed / accel
        self.d_ramp = 0.5 * accel * self.t_ramp ** 2
        if 2.0 * self.d_ramp <= length:
            self.triangular = False
            self.d_cruise = length - 2.0 * self.d_ramp
            self.t_cruise = self.d_cruise / speed
        else:
            self.triangular = True
            self.v = math.sqrt(accel * length)
            self.t_ramp = self.v / accel
            self.d_ramp = 0.5 * length
            self.d_cruise = self.t_cruise = 0.0
        self.t_total = 2.0 * self.t_ramp + self.t_cruise

    def at(self, t):
        a, v, tr = self.a, self.v, self.t_ramp
        if t <= 0.0:
            return 0.0, 0.0
        if t < tr:
            return 0.5 * a * t * t, a * t
        if t < tr + self.t_cruise:
            return self.d_ramp + v * (t - tr), v
        td = t - tr - self.t_cruise
        if td < tr:
            return (self.d_ramp + self.d_cruise + v * td - 0.5 * a * td * td,
                    v - a * td)
        return self.length, 0.0

    def describe(self):
        if self.triangular:
            return (f"TRIANGULAR: {self.length:.2f} m is too short for "
                    f"{self.v:.2f} m/s at {self.a:.2f} m/s^2")
        return (f"{self.length:.2f} m: ramp {self.d_ramp:.2f} m, "
                f"cruise {self.d_cruise:.2f} m at {self.v:.2f} m/s, "
                f"total {self.t_total:.1f} s, constant-speed window "
                f"{self.t_ramp:.1f}-{self.t_ramp + self.t_cruise:.1f} s")


class PanelCmd(Node):

    def __init__(self, a):
        super().__init__("panel_cmd")
        self.a = a
        self.panel = None          # (origin, n, t) once locked
        self.drone = None
        self.start = None
        self.t0 = None
        self.prof = None
        self.announced = False

        self.pub = self.create_publisher(PoseStamped, a.setpoint_topic, QOS)
        self.create_subscription(PoseStamped, a.panel_topic,
                                 self.on_panel, QOS)
        self.create_subscription(PoseWithCovarianceStamped, a.pose_topic,
                                 self.on_drone, QOS)
        self.create_timer(1.0 / a.rate, self.on_timer)
        self.get_logger().info(
            f"waiting for {a.panel_topic} and {a.pose_topic} ...")

    # -- inputs -------------------------------------------------------------

    def on_panel(self, msg: PoseStamped) -> None:
        if self.panel is not None:
            return                                   # lock the panel once
        o, q = msg.pose.position, msg.pose.orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        if not self.a.no_conj:                       # OptiTrack convention
            qx, qy, qz = -qx, -qy, -qz
        R = quat_to_R(qx, qy, qz, qw)

        col = {"x": 0, "y": 1, "z": 2}[self.a.normal_axis]
        sgn = -1.0 if self.a.flip_normal else 1.0
        n = unit([sgn * R[i][col] for i in range(3)])
        t = unit([-n[1], n[0], 0.0])                 # z_world x n, horizontal

        tilt = math.degrees(math.asin(max(-1.0, min(1.0, abs(n[2])))))
        origin = [o.x + self.a.offset_t * t[0],
                  o.y + self.a.offset_t * t[1],
                  o.z + self.a.offset_t * t[2] + self.a.offset_z]
        self.panel = (origin, n, t)
        self.get_logger().warning(
            f"panel locked: rb origin [{o.x:+.3f} {o.y:+.3f} {o.z:+.3f}]  "
            f"sheet centre [{origin[0]:+.3f} {origin[1]:+.3f} {origin[2]:+.3f}]  "
            f"normal [{n[0]:+.3f} {n[1]:+.3f} {n[2]:+.3f}]  "
            f"tangent [{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}]  "
            f"normal tilt from horizontal {tilt:.1f} deg")
        if tilt > self.a.max_tilt:
            self.get_logger().error(
                f"normal is {tilt:.1f} deg off horizontal, --max-tilt is "
                f"{self.a.max_tilt}. Wrong axis? Try --flip-normal, or check "
                f"the rigid body definition in Motive.")
            rclpy.shutdown()

    def on_drone(self, msg: PoseWithCovarianceStamped) -> None:
        p = msg.pose.pose.position
        self.drone = [p.x, p.y, p.z]

    # -- geometry -----------------------------------------------------------

    def target_at(self, lateral):
        o, n, t = self.panel
        d, dz = self.a.standoff, self.a.dz
        return [o[i] + d * n[i] + lateral * t[i] + (dz if i == 2 else 0.0)
                for i in range(3)]

    def yaw(self):
        _, n, _ = self.panel
        return math.atan2(-n[1], -n[0])

    # -- output -------------------------------------------------------------

    def on_timer(self) -> None:
        if self.panel is None or self.drone is None:
            return

        if self.a.report:
            self.report_only()
            return

        if self.a.goto:
            self.publish(self.target_at(self.a.lateral))
            if not self.announced:
                tgt = self.target_at(self.a.lateral)
                self.get_logger().warning(
                    f"holding [{tgt[0]:+.3f} {tgt[1]:+.3f} {tgt[2]:+.3f}] "
                    f"yaw {math.degrees(self.yaw()):+.1f} deg  "
                    f"(move {math.dist(self.drone, tgt):.2f} m)")
                self.announced = True
            return

        # pass mode
        if self.start is None:
            half = self.a.length / 2.0
            s0 = -half if self.a.dir > 0 else half
            self.start = self.target_at(self.a.lateral + s0)
            gap = math.dist(self.drone, self.start)
            if gap > self.a.max_jump:
                self.get_logger().error(
                    f"pass start is {gap:.2f} m away, --max-jump is "
                    f"{self.a.max_jump}. Use --goto --lateral "
                    f"{self.a.lateral + s0:+.2f} first.")
                rclpy.shutdown()
                return
            self.prof = Profile(self.a.length, self.a.speed, self.a.accel)
            if self.prof.triangular:
                self.get_logger().error(self.prof.describe())
                rclpy.shutdown()
                return
            self.t0 = time.monotonic()
            self.get_logger().warning(
                f"pass from [{self.start[0]:+.2f} {self.start[1]:+.2f} "
                f"{self.start[2]:+.2f}] along {'+t' if self.a.dir > 0 else '-t'}"
                f"\n  {self.prof.describe()}"
                f"\n  lead at cruise {self.a.speed / self.a.kp_xy:.2f} m")

        t = time.monotonic() - self.t0
        s, v = self.prof.at(t)
        r = s + v / self.a.kp_xy
        if self.a.clamp:
            r = min(r, self.prof.length)
        sgn = 1.0 if self.a.dir > 0 else -1.0
        _, _, tan = self.panel
        self.publish([self.start[i] + sgn * r * tan[i] for i in range(3)])

        if not self.announced and t >= self.prof.t_total:
            self.announced = True
            self.get_logger().warning("pass complete, holding at the end")

    def publish(self, p):
        w = self.yaw()
        m = PoseStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.a.frame_id
        m.pose.position.x = float(p[0])
        m.pose.position.y = float(p[1])
        m.pose.position.z = float(p[2])
        m.pose.orientation.z = float(math.sin(w / 2.0))
        m.pose.orientation.w = float(math.cos(w / 2.0))
        self.pub.publish(m)

    def report_only(self):
        o, n, t = self.panel
        print(f"\npanel origin  [{o[0]:+.3f} {o[1]:+.3f} {o[2]:+.3f}]")
        print(f"panel normal  [{n[0]:+.3f} {n[1]:+.3f} {n[2]:+.3f}]")
        print(f"pass tangent  [{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}]")
        print(f"facing yaw    {math.degrees(self.yaw()):+.1f} deg")
        half = self.a.length / 2.0
        for name, lat in (("start (-t end)", -half), ("centre", 0.0),
                          ("end   (+t end)", half)):
            p = self.target_at(lat)
            print(f"  {name:<15} [{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}]  "
                  f"drone is {math.dist(self.drone, p):.2f} m away")
        rclpy.shutdown()


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--goto', action='store_true')
    mode.add_argument('--pass', dest='do_pass', action='store_true')
    mode.add_argument('--report', action='store_true')

    p.add_argument('--standoff', type=float, required=True, help='m along +n')
    p.add_argument('--lateral', type=float, default=0.0,
                   help='m along the tangent from panel centre')
    p.add_argument('--dz', type=float, default=0.0,
                   help='m above the panel origin')
    p.add_argument('--length', type=float, default=3.0, help='pass length, m')
    p.add_argument('--speed', type=float, default=0.5, help='m/s')
    p.add_argument('--accel', type=float, default=0.5, help='m/s^2')
    p.add_argument('--dir', type=int, default=1, choices=(1, -1))
    p.add_argument('--kp-xy', type=float, default=0.6)

    p.add_argument('--panel-topic',
                   default='/optitrack/rigid_bodies/panel')
    p.add_argument('--pose-topic', default='/drone_1/localisation/pose')
    p.add_argument('--setpoint-topic', default='/drone_1/setpoint')
    p.add_argument('--frame-id', default='odom')
    p.add_argument('--rate', type=float, default=20.0)

    p.add_argument('--normal-axis', default='z', choices=('x', 'y', 'z'),
                   help='axis of the panel rigid body normal to the printed '
                        'face. Motive often creates rigid bodies '
                        'world-aligned, so check with --report before flying.')
    p.add_argument('--offset-t', type=float, default=0.0,
                   help='m along the tangent, rigid body origin -> sheet '
                        'centre. A4 sheet, origin at a side edge: 0.105')
    p.add_argument('--offset-z', type=float, default=0.0,
                   help='m vertically, rigid body origin -> sheet centre. '
                        'A4 sheet, origin at the top edge: -0.1485')
    p.add_argument('--flip-normal', action='store_true',
                   help='negate the axis, so it points towards the aircraft')
    p.add_argument('--no-conj', action='store_true')
    p.add_argument('--no-clamp', dest='clamp', action='store_false')
    p.add_argument('--max-tilt', type=float, default=20.0,
                   help='deg, refuse if the panel normal is not horizontal')
    p.add_argument('--max-jump', type=float, default=0.5,
                   help='m, refuse to start a pass from further than this')
    p.add_argument('--max-speed', type=float, default=1.0)
    a = p.parse_args()

    if a.speed > a.max_speed:
        sys.exit(f"refusing {a.speed:.2f} m/s, --max-speed {a.max_speed}")

    rclpy.init()
    node = PanelCmd(a)
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
