#!/usr/bin/env python3
"""H.264 TCP -> sensor_msgs/Image, decoded in-process with PyAV.
"""
import fractions
import time

import av
import av.error
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class H264TcpDecode(Node):
    def __init__(self):
        super().__init__("h264_tcp_decode")

        # parameters
        self.declare_parameter("host", "192.168.1.100")   # phone IP on the LAN
        self.declare_parameter("port", 8900)
        self.declare_parameter("topic", "/camera/image_raw")
        self.declare_parameter("frame_id", "camera_link")
        self.declare_parameter("reconnect_backoff_s", 2.0)
        self.declare_parameter("encoding", "bgr8")        # cv_bridge output encoding
        # Probe sizing for av.open(): large enough to find SPS/PPS + an IDR so the
        # open returns a configured decoder, not a partial slice.
        self.declare_parameter("probesize", 5000000)        # bytes
        self.declare_parameter("analyzeduration", 2000000)  # microseconds
        self.declare_parameter("open_timeout_s", 30.0)
        # Stamping: PTS-based (preferred for localization) vs node arrival time.
        # PTS is used only when available; otherwise we fall back to arrival time.
        self.declare_parameter("use_pts", False)

        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        topic = self.get_parameter("topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.backoff = float(self.get_parameter("reconnect_backoff_s").value)
        self.encoding = self.get_parameter("encoding").value
        self.probesize = int(self.get_parameter("probesize").value)
        self.analyzeduration = int(self.get_parameter("analyzeduration").value)
        self.open_timeout = float(self.get_parameter("open_timeout_s").value)
        self.use_pts = bool(self.get_parameter("use_pts").value)
        # ROS image encoding -> FFmpeg/PyAV pixel format name
        self._pix_fmt = {"bgr8": "bgr24", "rgb8": "rgb24", "mono8": "gray"}.get(
            self.encoding, "bgr24"
        )

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, topic, 10)

        self.get_logger().info(
            f"h264_tcp_decode -> {topic} (frame_id={self.frame_id}) "
            f"source tcp://{self.host}:{self.port} "
            f"stamp={'pts' if self.use_pts else 'arrival'}"
        )

    def run(self):
        """Blocking connect/decode/publish loop with reconnect."""
        while rclpy.ok():
            try:
                n = self._stream_once()
                self.get_logger().warn(
                    f"stream ended after {n} frame(s); reconnecting in {self.backoff}s"
                )
            except Exception as exc:  # noqa: BLE001 - log and retry any stream error
                self.get_logger().warn(
                    f"stream error: {exc}; reconnecting in {self.backoff}s"
                )
            if rclpy.ok():
                time.sleep(self.backoff)

    def _stream_once(self):
        url = f"tcp://{self.host}:{self.port}"
        self.get_logger().info(f"connecting to {url}")
        # nobuffer/low_delay for live latency; generous probe so the open lands
        # on a configured decoder. No read timeout option here: a live stream
        # delivers in bursts and an eager timeout can make demux hit EOF early.
        options = {
            "fflags": "nobuffer",
            "flags": "low_delay",
            "probesize": str(self.probesize),
            "analyzeduration": str(self.analyzeduration),
        }
        container = av.open(
            url, format="h264", mode="r", timeout=self.open_timeout, options=options
        )
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = stream.time_base
        self.get_logger().info("connected; decoding")

        count = 0
        got_key = False
        try:
            # Demux ourselves so a per-packet decode failure (a partial/garbage
            # slice before the first keyframe) is recoverable -- keep reading like
            # ffplay instead of ending the stream.
            for packet in container.demux(stream):
                if not rclpy.ok():
                    break
                try:
                    frames = packet.decode()
                except (av.error.InvalidDataError, av.error.ValueError):
                    continue
                for frame in frames:
                    if not got_key:
                        if not frame.key_frame:
                            continue          # wait for first IDR
                        got_key = True
                        self.get_logger().info("first keyframe decoded; publishing")
                    img = frame.to_ndarray(format=self._pix_fmt)
                    msg = self.bridge.cv2_to_imgmsg(img, encoding=self.encoding)
                    msg.header.stamp = self._stamp(frame, time_base)
                    msg.header.frame_id = self.frame_id
                    self.pub.publish(msg)
                    count += 1
        finally:
            container.close()
        return count

    def _stamp(self, frame, time_base):
        """ROS time for a frame: PTS-derived if requested and available, else now.

        NOTE: PTS here is on the stream clock, not yet aligned to the ROS/system
        or telemetry clock. Aligning the two (the common-time-base work) is still
        TODO; until then use_pts gives relative timing only. Arrival time is the
        safe default.
        """
        if self.use_pts and frame.pts is not None and time_base is not None:
            t = float(frame.pts * fractions.Fraction(time_base))  # seconds
            stamp = self.get_clock().now().to_msg()
            # Replace with PTS-derived seconds once a shared epoch is defined.
            stamp.sec = int(t)
            stamp.nanosec = int((t - int(t)) * 1e9)
            return stamp
        return self.get_clock().now().to_msg()  # arrival time


def main(args=None):
    rclpy.init(args=args)
    node = H264TcpDecode()
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