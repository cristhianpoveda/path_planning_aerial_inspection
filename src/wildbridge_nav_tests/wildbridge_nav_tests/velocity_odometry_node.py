import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Vector3, Point

class VelocityOdometryNode(Node):
    def __init__(self) -> None:
        super().__init__('velocity_odometry_node')
        
        # --- Subscribers & Publishers ---
        self.sub_speed = self.create_subscription(
            Vector3, 
            '/drone_1/speed_vector', 
            self.speed_callback, 
            10
        )
        
        self.pub_position = self.create_publisher(
            Point, 
            '/drone_1/cartesian_position', 
            10
        )

        # --- State Variables ---
        self.current_pos = Point()
        self.current_pos.x = 0.0
        self.current_pos.y = 0.0
        self.current_pos.z = 0.0
        
        self.last_time_ns: int = 0
        self.is_initialized: bool = False
        
        self.get_logger().info("Velocity Odometry initialized. Integrating /drone_1/speed_vector...")

    def speed_callback(self, msg: Vector3) -> None:
        """Triggered every time a new speed packet arrives."""
        current_time_ns = self.get_clock().now().nanoseconds
        
        if not self.is_initialized:
            self.last_time_ns = current_time_ns
            self.is_initialized = True
            return

        # Calculate exact elapsed time in seconds
        dt = (current_time_ns - self.last_time_ns) / 1e9
        
        # Integrate Velocity to get Position (P = P + v*dt)
        self.current_pos.x += msg.x * dt
        self.current_pos.y += msg.y * dt
        self.current_pos.z += msg.z * dt
        
        # Update timestamp for the next cycle
        self.last_time_ns = current_time_ns

        # Publish the estimated local position
        self.pub_position.publish(self.current_pos)

def main(args=None):
    rclpy.init(args=args)
    node = VelocityOdometryNode()
    
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()