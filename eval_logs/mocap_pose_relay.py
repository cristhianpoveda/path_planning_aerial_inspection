#!/usr/bin/env python3
"""mocap_pose_relay -- OptiTrack pose as controller feedback.

Republishes the mocap rigid body as `localisation/pose` so the position
controller can be tuned against a perfect estimate, before the EKF is put in
the loop. Run in the same ROS domain as OptiTrack (0); no domain bridge.

    python3 mocap_pose_relay.py
    python3 mocap_pose_relay.py --yaw-offset 0.5 --rate 30
"""
import argparse
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=10)


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


class MocapRelay(Node):

    def __init__(self, a):
        super().__init__("mocap_pose_relay")
        self.a = a
        self.conj = not a.no_conj
        self.yaw_q = (0.0, 0.0, math.sin(math.radians(a.yaw_offset) / 2.0),
                      math.cos(math.radians(a.yaw_offset) / 2.0))
        self.min_dt = 1.0 / a.rate if a.rate > 0 else 0.0

        self._last_key = None
        self._repeats = 0
        self._t_last_pub = 0.0
        self._n_in = 0
        self._n_out = 0
        self._n_drop = 0
        self._warned = False

        self.pub = self.create_publisher(
            PoseWithCovarianceStamped, a.out_topic, QOS)
        self.create_subscription(PoseStamped, a.in_topic, self.on_pose, QOS)

        self.status_pub = None
        if not a.no_status:
            try:
                from drone_interfaces.msg import LocalisationStatus
                self._LS = LocalisationStatus
                self.status_pub = self.create_publisher(
                    LocalisationStatus, a.status_topic, QOS)
            except ImportError:
                self.get_logger().warning(
                    "drone_interfaces not importable; not publishing status. "
                    "Run the controller with gate_on_degraded:=false")

        self.create_timer(5.0, self.report)
        self.get_logger().info(
            f"relay {a.in_topic} -> {a.out_topic}  "
            f"conjugate={self.conj}  yaw_offset={a.yaw_offset} deg  "
            f"max_rate={a.rate} Hz  stale_after={a.stale_repeats} repeats")

    def on_pose(self, msg: PoseStamped) -> None:
        self._n_in += 1
        p, o = msg.pose.position, msg.pose.orientation
        key = (p.x, p.y, p.z, o.x, o.y, o.z, o.w)

        # Repeated poses mean the rigid body is not being tracked. Publishing
        # them would look like a healthy stationary aircraft.
        if key == self._last_key:
            self._repeats += 1
            self._n_drop += 1
            if self._repeats == self.a.stale_repeats and not self._warned:
                self.get_logger().error(
                    f"rigid body repeated {self._repeats} times -- tracking "
                    f"lost. Output stopped; the controller will time out.")
                self._warned = True
            return
        self._last_key = key
        if self._warned:
            self.get_logger().warning("tracking reacquired")
        self._repeats = 0
        self._warned = False

        now = self.get_clock().now().nanoseconds * 1e-9
        if self.min_dt and now - self._t_last_pub < self.min_dt:
            return
        self._t_last_pub = now

        q = (o.x, o.y, o.z, o.w)
        if self.conj:
            q = (-q[0], -q[1], -q[2], q[3])
        if self.a.yaw_offset != 0.0:
            q = quat_mul(self.yaw_q, q)

        out = PoseWithCovarianceStamped()
        out.header = msg.header
        out.pose.pose.position = p
        out.pose.pose.orientation.x = float(q[0])
        out.pose.pose.orientation.y = float(q[1])
        out.pose.pose.orientation.z = float(q[2])
        out.pose.pose.orientation.w = float(q[3])
        c = [0.0] * 36
        for i, v in enumerate([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]):
            c[i * 6 + i] = v
        out.pose.covariance = c
        self.pub.publish(out)
        self._n_out += 1

        if self.status_pub is not None:
            st = self._LS()
            st.header = msg.header
            st.state = "OK"
            st.scale = 1.0
            st.sigma_scale = 0.0
            st.alt_bias = 0.0
            st.sigma_alt_bias = 0.0
            st.cov_scale_bias = 0.0
            st.degraded = False
            st.flags = []
            self.status_pub.publish(st)

    def report(self) -> None:
        self.get_logger().info(
            f"in={self._n_in} out={self._n_out} dropped_repeats={self._n_drop}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in-topic', default='/optitrack/rigid_bodies/dji_mini4')
    p.add_argument('--out-topic', default='/drone_1/localisation/pose')
    p.add_argument('--status-topic', default='/drone_1/localisation/status')
    p.add_argument('--rate', type=float, default=30.0,
                   help='max republish rate, Hz; 0 = every message')
    p.add_argument('--yaw-offset', type=float, default=0.0,
                   help='deg, mocap rigid body -> base_link')
    p.add_argument('--stale-repeats', type=int, default=10)
    p.add_argument('--no-conj', action='store_true')
    p.add_argument('--no-status', action='store_true')
    a = p.parse_args()

    rclpy.init()
    node = MocapRelay(a)
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
    