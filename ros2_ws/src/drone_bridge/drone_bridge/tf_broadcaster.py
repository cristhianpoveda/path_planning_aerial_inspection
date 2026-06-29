#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

# Check the actual type with: ros2 topic info /optitrack/rigid_bodies/dji_mini4
from mocap4r2_msgs.msg import RigidBodies


class DjiMini4TFBroadcaster(Node):
    def __init__(self):
        super().__init__('dji_mini4_tf_broadcaster')

        # Dynamic broadcaster for the OptiTrack pose
        self.tf_broadcaster = TransformBroadcaster(self)

        # Static broadcaster for fixed offsets (sent once at startup)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_transforms()

        self.sub = self.create_subscription(
            RigidBodies,
            '/optitrack/rigid_bodies/dji_mini4',
            self.optitrack_callback,
            10)

    def _publish_static_transforms(self):
        now = self.get_clock().now().to_msg()

        # 1. marker → body
        # Measure this with a ruler on the physical drone.
        # Example: body centre is 5cm forward and 1cm up from the back marker.
        # Axes follow ROS convention: x=forward, y=left, z=up
        t_marker_body = TransformStamped()
        t_marker_body.header.stamp = now
        t_marker_body.header.frame_id = 'dji_mini4/optitrack_marker'
        t_marker_body.child_frame_id = 'dji_mini4/body'
        t_marker_body.transform.translation.x = -0.045  # forward (m)
        t_marker_body.transform.translation.y = -0.023
        t_marker_body.transform.translation.z = -0.086   # upward (m)
        t_marker_body.transform.rotation.x = 0.0
        t_marker_body.transform.rotation.y = 0.0
        t_marker_body.transform.rotation.z = 0.0
        t_marker_body.transform.rotation.w = 1.0

        # 2. body → camera
        # Replace with your actual calibrated extrinsic values.
        t_body_camera = TransformStamped()
        t_body_camera.header.stamp = now
        t_body_camera.header.frame_id = 'dji_mini4/body'
        t_body_camera.child_frame_id = 'dji_mini4/camera'
        t_body_camera.transform.translation.x = 0.06   # forward from body centre
        t_body_camera.transform.translation.y = 0.0
        t_body_camera.transform.translation.z = 0.38  # below body centre
        t_body_camera.transform.rotation.x = 0.0
        t_body_camera.transform.rotation.y = 0.0
        t_body_camera.transform.rotation.z = 0.0
        t_body_camera.transform.rotation.w = 1.0

        self.static_broadcaster.sendTransform(
            [t_marker_body, t_body_camera])

    def optitrack_callback(self, msg):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()   # use OptiTrack's own timestamp
        t.header.frame_id = 'origin'
        t.child_frame_id = 'dji_mini4/optitrack_marker'
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = DjiMini4TFBroadcaster()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()