#!/usr/bin/env python3
"""comparison_node — record localisation estimate + OptiTrack ground truth to TUM.

Buffers both pose streams while recording, and writes them as TUM-format
trajectory files for offline analysis with evo on the host:

    evo_ape tum gt_<stamp>.tum est_<stamp>.tum -va --plot
    evo_rpe tum gt_<stamp>.tum est_<stamp>.tum -va --delta 1 --delta_unit m

Recording is gated by two std_srvs/Trigger services:
    ~/start  — clear buffers, begin recording
    ~/stop   — stop recording, write both .tum files, return the paths

TUM format (one pose per line, space separated):
    timestamp tx ty tz qx qy qz qw
"""
import os
from datetime import datetime

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
import math


class ComparisonNode(Node):
    def __init__(self):
        super().__init__("comparison_node")

        self.declare_parameter("log_path", "/eval_logs")
        self.declare_parameter("mocap_frame", "optitrack_map")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("mocap_translation", [0.0, 0.0, 0.0])    # x, y, z [m]
        self.declare_parameter("mocap_rotation_rpy", [0.0, 0.0, 0.0])   # roll, pitch, yaw [rad]

        self.log_path = self.get_parameter("log_path").get_parameter_value().string_value
        

        self._recording = False

        # Buffers of TUM rows: (t, x, y, z, qx, qy, qz, qw)
        self._est_rows = []
        self._gt_rows = []

        self._static_tf = StaticTransformBroadcaster(self)
        self._publish_mocap_transform()

        self.create_subscription(
            PoseWithCovarianceStamped, "localisation/pose", self._on_estimate, 10)
        self.create_subscription(
            PoseStamped, "/optitrack/rigid_bodies/dji_mini4", self._on_ground_truth, 10)

        self.create_service(Trigger, "~/start", self._srv_start)
        self.create_service(Trigger, "~/stop", self._srv_stop)

        self.get_logger().info(
            f"comparison_node ready"
            f"out='{self.log_path}'"
        )

    def _publish_mocap_transform(self):
        """optitrack_map -> map : manual alignment, for visualization only.
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.get_parameter("mocap_frame").value
        t.child_frame_id = self.get_parameter("map_frame").value

        tr = self.get_parameter("mocap_translation").get_parameter_value().double_array_value
        rpy = self.get_parameter("mocap_rotation_rpy").get_parameter_value().double_array_value

        t.transform.translation.x = tr[0]
        t.transform.translation.y = tr[1]
        t.transform.translation.z = tr[2]

        cr, sr = math.cos(rpy[0] * 0.5), math.sin(rpy[0] * 0.5)
        cp, sp = math.cos(rpy[1] * 0.5), math.sin(rpy[1] * 0.5)
        cy, sy = math.cos(rpy[2] * 0.5), math.sin(rpy[2] * 0.5)
        t.transform.rotation.x = sr * cp * cy - cr * sp * sy
        t.transform.rotation.y = cr * sp * cy + sr * cp * sy
        t.transform.rotation.z = cr * cp * sy - sr * sp * cy
        t.transform.rotation.w = cr * cp * cy + sr * sp * sy

        self._static_tf.sendTransform(t)
        self.get_logger().info(
            f"static tf {t.header.frame_id} -> {t.child_frame_id} (visualization only)")

    # --- callbacks
    @staticmethod
    def _row(header, pose):
        t = header.stamp.sec + header.stamp.nanosec * 1e-9
        p, q = pose.position, pose.orientation
        return (t, p.x, p.y, p.z, q.x, q.y, q.z, q.w)

    def _on_estimate(self, msg: PoseWithCovarianceStamped):
        if self._recording:
            self._est_rows.append(self._row(msg.header, msg.pose.pose))

    def _on_ground_truth(self, msg: PoseStamped):
        if self._recording:
            self._gt_rows.append(self._row(msg.header, msg.pose))

    # --- services
    def _srv_start(self, request, response):
        self._est_rows.clear()
        self._gt_rows.clear()
        self._recording = True
        response.success = True
        response.message = "recording started"
        self.get_logger().info(response.message)
        return response

    def _srv_stop(self, request, response):
        if not self._recording:
            response.success = False
            response.message = "not recording"
            self.get_logger().warn(response.message)
            return response

        self._recording = False
        n_est, n_gt = len(self._est_rows), len(self._gt_rows)

        if n_est == 0 or n_gt == 0:
            response.success = False
            response.message = f"nothing to write (est={n_est}, gt={n_gt})"
            self.get_logger().warn(response.message)
            return response

        paths = self._write_tum()
        response.success = True
        response.message = (
            f"wrote est({n_est}) -> {paths[0]} ; gt({n_gt}) -> {paths[1]}"
        )
        self.get_logger().info(response.message)
        return response

    # --- output
    def _write_tum(self):
        os.makedirs(self.log_path, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        est_path = os.path.join(self.log_path, f"est_{stamp}.tum")
        gt_path = os.path.join(self.log_path, f"gt_{stamp}.tum")

        for path, rows in ((est_path, self._est_rows), (gt_path, self._gt_rows)):
            with open(path, "w") as f:
                for r in rows:
                    f.write(
                        f"{r[0]:.9f} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} "
                        f"{r[4]:.6f} {r[5]:.6f} {r[6]:.6f} {r[7]:.6f}\n"
                    )
        return est_path, gt_path


def main(args=None):
    rclpy.init(args=args)
    node = ComparisonNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    finally:
        
        if node._recording and node._est_rows and node._gt_rows:
            node._write_tum()
            node.get_logger().info("flushed buffers on shutdown")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
