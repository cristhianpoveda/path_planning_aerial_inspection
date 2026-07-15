"""ekf_node — skeleton stub.

Starts a ROS 2 node with no behaviour yet. Topics, parameters and logic are
added when this node is implemented (see node_interfaces.md).
"""
import rclpy
from rclpy.node import Node


class EkfNode(Node):
    def __init__(self) -> None:
        super().__init__("ekf_node")
        self.get_logger().info("ekf_node skeleton up")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EkfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
