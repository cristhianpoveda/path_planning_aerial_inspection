#!/usr/bin/env python3
"""setpoint_cmd -- publish a controller setpoint, relative or absolute.

Latches the aircraft's CURRENT pose at startup, applies the requested offset
once, and republishes the result at a fixed rate until Ctrl-C. Latching means
the setpoint never chases the aircraft, and republishing keeps the
controller's setpoint_timeout satisfied.

    python3 setpoint_cmd.py --hold              # hold where you are
    python3 setpoint_cmd.py --dx 0.3            # 0.3 m forward in odom/mocap x
    python3 setpoint_cmd.py --dz 0.3 --dyaw 20  # climb and turn
    python3 setpoint_cmd.py --x 1.0 --y 0.0 --z 1.5   # absolute

Offsets are in the FEEDBACK frame (mocap or odom), not the body frame.
"""
import argparse
import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=10)


def quat_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class SetpointCmd(Node):

    def __init__(self, a):
        super().__init__("setpoint_cmd")
        self.a = a
        self.target = None
        self.pub = self.create_publisher(PoseStamped, a.setpoint_topic, QOS)
        self.create_subscription(PoseWithCovarianceStamped, a.pose_topic,
                                 self.on_pose, QOS)
        self.create_timer(1.0 / a.rate, self.on_timer)
        self.get_logger().info(f"waiting for {a.pose_topic} ...")

    def on_pose(self, msg: PoseWithCovarianceStamped) -> None:
        if self.target is not None:
            return                       # latch once
        p = msg.pose.pose.position
        yaw = quat_yaw(msg.pose.pose.orientation)
        a = self.a
        x = a.x if a.x is not None else p.x + a.dx
        y = a.y if a.y is not None else p.y + a.dy
        z = a.z if a.z is not None else p.z + a.dz
        w = (math.radians(a.yaw) if a.yaw is not None
             else yaw + math.radians(a.dyaw))
        self.target = (x, y, z, w)
        self.get_logger().warning(
            f"latched: current [{p.x:+.3f} {p.y:+.3f} {p.z:+.3f}] "
            f"yaw {math.degrees(yaw):+.1f} deg  ->  "
            f"target [{x:+.3f} {y:+.3f} {z:+.3f}] "
            f"yaw {math.degrees(w):+.1f} deg  "
            f"(move {math.dist((p.x, p.y, p.z), (x, y, z)):.3f} m)")

    def on_timer(self) -> None:
        if self.target is None:
            return
        x, y, z, w = self.target
        m = PoseStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.a.frame_id
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.pose.orientation.z = float(math.sin(w / 2.0))
        m.pose.orientation.w = float(math.cos(w / 2.0))
        self.pub.publish(m)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dx', type=float, default=0.0)
    p.add_argument('--dy', type=float, default=0.0)
    p.add_argument('--dz', type=float, default=0.0)
    p.add_argument('--dyaw', type=float, default=0.0, help='deg')
    p.add_argument('--x', type=float, default=None, help='absolute, overrides dx')
    p.add_argument('--y', type=float, default=None)
    p.add_argument('--z', type=float, default=None)
    p.add_argument('--yaw', type=float, default=None, help='deg, absolute')
    p.add_argument('--hold', action='store_true', help='same as no offset')
    p.add_argument('--pose-topic', default='/drone_1/localisation/pose')
    p.add_argument('--setpoint-topic', default='/drone_1/setpoint')
    p.add_argument('--frame-id', default='odom')
    p.add_argument('--rate', type=float, default=5.0)
    p.add_argument('--max-step', type=float, default=2.0,
                   help='refuse offsets larger than this, m')
    a = p.parse_args()

    step = math.sqrt(a.dx ** 2 + a.dy ** 2 + a.dz ** 2)
    if not a.hold and step > a.max_step:
        print(f"refusing a {step:.2f} m step, --max-step is {a.max_step}")
        sys.exit(1)

    rclpy.init()
    node = SetpointCmd(a)
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
    