#!/usr/bin/env python3
"""
Camera Decoder Node
"""
import math
import time

import av
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
import json
import threading
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
import cv2


class CameraDecoderNode(Node):
    def __init__(self):
        super().__init__("camera_decoder_node")

        # --- connection ---
        self.declare_parameter("host", "192.168.50.18")
        self.declare_parameter("port", 8900)
        self.declare_parameter("reconnect_backoff_s", 2.0)

        # --- ffmpeg / PyAV open options ---
        self.declare_parameter("fflags", "nobuffer")
        self.declare_parameter("flags", "low_delay")
        self.declare_parameter("analyzeduration", 0)   # microseconds
        self.declare_parameter("probesize", 32)        # bytes

        # --- frames + static extrinsics ---
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("gimbal_base_frame", "gimbal_base")
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("optical_frame", "camera_optical_frame")
        # base_link -> gimbal_base mount
        self.declare_parameter("mount_translation", [0.105, 0.0, -0.025])    # x, y, z [m]
        self.declare_parameter("mount_rotation_rpy", [0.0, 0.0, 0.0])   # roll, pitch, yaw [rad]

        self.host = self.get_parameter("host").value
        self.port = self.get_parameter("port").get_parameter_value().integer_value
        self.backoff = self.get_parameter("reconnect_backoff_s").get_parameter_value().double_value

        self.fflags = self.get_parameter("fflags").value
        self.flags = self.get_parameter("flags").value
        self.analyzeduration = self.get_parameter("analyzeduration").get_parameter_value().integer_value
        self.probesize = self.get_parameter("probesize").get_parameter_value().integer_value

        self.base_frame = self.get_parameter("base_frame").value
        self.gimbal_base_frame = self.get_parameter("gimbal_base_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.optical_frame = self.get_parameter("optical_frame").value
        self.mount_translation = self.get_parameter("mount_translation").get_parameter_value().double_array_value
        self.mount_rotation_rpy = self.get_parameter("mount_rotation_rpy").get_parameter_value().double_array_value

        self.bridge = CvBridge()
        
        self.pub = self.create_publisher(CompressedImage, "/camera/image/compressed", qos_profile_sensor_data)

        # Static (latched): base_link -> gimbal_base, camera_link -> optical.
        self._static_tf = StaticTransformBroadcaster(self)
        self._publish_static_transforms()

        # Dynamic: gimbal_base -> camera_link
        self._gimbal_tf = TransformBroadcaster(self)
        self.create_subscription(String, "gimbal_joint_attitude", self._on_gimbal_joint, 10)

        self.get_logger().info(
            f"camera_decoder_node -> /camera/image/compressed  "
            f"source tcp://{self.host}:{self.port}  "
            f"frames {self.base_frame}->{self.camera_frame}->{self.optical_frame}"
        )

    # --- transforms
    def _publish_static_transforms(self):
        now = self.get_clock().now().to_msg()

        # base_link -> gimbal_base : fixed mount offset (midpoint of the gimbal pivot).
        body_to_gimbal = TransformStamped()
        body_to_gimbal.header.stamp = now
        body_to_gimbal.header.frame_id = self.base_frame
        body_to_gimbal.child_frame_id = self.gimbal_base_frame
        body_to_gimbal.transform.translation.x = self.mount_translation[0]
        body_to_gimbal.transform.translation.y = self.mount_translation[1]
        body_to_gimbal.transform.translation.z = self.mount_translation[2]
        qx, qy, qz, qw = self._quat_from_euler(*self.mount_rotation_rpy)
        body_to_gimbal.transform.rotation.x = qx
        body_to_gimbal.transform.rotation.y = qy
        body_to_gimbal.transform.rotation.z = qz
        body_to_gimbal.transform.rotation.w = qw

        # camera_link -> camera_optical_frame : fixed optical convention (REP 103).
        cam_to_optical = TransformStamped()
        cam_to_optical.header.stamp = now
        cam_to_optical.header.frame_id = self.camera_frame
        cam_to_optical.child_frame_id = self.optical_frame
        cam_to_optical.transform.rotation.x = -0.5
        cam_to_optical.transform.rotation.y = 0.5
        cam_to_optical.transform.rotation.z = -0.5
        cam_to_optical.transform.rotation.w = 0.5

        self._static_tf.sendTransform([body_to_gimbal, cam_to_optical])

    @staticmethod
    def _quat_from_euler(roll, pitch, yaw):
        """RPY (ZYX, tf2 convention) -> quaternion (x, y, z, w)."""
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        return (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )

    # --- decode
    def run(self):
        """Connect, decode, publish; reconnect on any error."""
        while rclpy.ok():
            try:
                self._stream_once()
            except Exception as exc:  # log and retry
                self.get_logger().warn(f"stream error: {exc}; reconnecting in {self.backoff}s")
            if rclpy.ok():
                time.sleep(self.backoff)

    def _on_gimbal_joint(self, msg):
        """gimbal_base -> camera_link from live joint angles (degrees in the topic)."""
        try:
            j = json.loads(msg.data.replace("'", '"'))
            pitch_deg = float(j["pitch"])
            if pitch_deg > 3276.75:
                pitch_deg -= 6553.5
            roll = math.radians(float(j["roll"]))
            pitch = math.radians(-pitch_deg)
            yaw = math.radians(-float(j["yaw"]))
        except (ValueError, KeyError) as exc:
            self.get_logger().warn(f"bad gimbal_joint msg: {exc}")
            return

        qx, qy, qz, qw = self._quat_from_euler(roll, pitch, yaw)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.gimbal_base_frame
        t.child_frame_id = self.camera_frame
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self._gimbal_tf.sendTransform(t)

    def _stream_once(self):
        url = f"tcp://{self.host}:{self.port}"
        self.get_logger().info(f"connecting to {url}")

        options = {
            "fflags": self.fflags,
            "flags": self.flags,
            "analyzeduration": str(self.analyzeduration),
            "probesize": str(self.probesize),
        }
        container = av.open(url, format="h264", mode="r", options=options)
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        self.get_logger().info("connected; decoding")

        got_key = False
        try:
            for frame in container.decode(stream):
                if not rclpy.ok():
                    break
                if not got_key:
                    if not frame.key_frame:
                        continue          # wait for the first keyframe
                    got_key = True
                img = frame.to_ndarray(format="bgr24")
                msg = self.bridge.cv2_to_compressed_imgmsg(img)
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.optical_frame
                self.pub.publish(msg)
        finally:
            container.close()


def main(args=None):
    rclpy.init(args=args)
    node = CameraDecoderNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()