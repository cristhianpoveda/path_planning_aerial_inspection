#!/usr/bin/env python3
"""yaw_monitor -- print yaw, roll, pitch and position from localisation/pose.

For the ground-based yaw sign check: hold the aircraft level, watch the YAW
column, and rotate it counter-clockwise seen from above. Yaw must INCREASE.

    python3 yaw_monitor.py
    python3 yaw_monitor.py --topic /drone_1/localisation/pose --rate 5
"""
import argparse
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseWithCovarianceStamped

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=10)


def rpy(q):
    """roll, pitch, yaw in degrees from (x, y, z, w)."""
    x, y, z, w = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class YawMonitor(Node):

    def __init__(self, a):
        super().__init__("yaw_monitor")
        self.msg = None
        self.n = 0
        self.create_subscription(PoseWithCovarianceStamped, a.topic,
                                 self.on_pose, QOS)
        self.create_timer(1.0 / a.rate, self.on_timer)
        print(f"listening on {a.topic}\n")
        print(f"{'x':>8}{'y':>8}{'z':>8}   "
              f"{'roll':>8}{'pitch':>8}{'YAW':>9}")

    def on_pose(self, msg):
        self.msg = msg
        self.n += 1

    def on_timer(self):
        if self.msg is None:
            print("  waiting for pose ...", end="\r")
            return
        p = self.msg.pose.pose.position
        o = self.msg.pose.pose.orientation
        r, pt, y = rpy((o.x, o.y, o.z, o.w))
        print(f"{p.x:8.3f}{p.y:8.3f}{p.z:8.3f}   "
              f"{r:8.1f}{pt:8.1f}{y:9.1f}", end="\r", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/drone_1/localisation/pose')
    ap.add_argument('--rate', type=float, default=5.0)
    a = ap.parse_args()
    rclpy.init()
    node = YawMonitor(a)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
