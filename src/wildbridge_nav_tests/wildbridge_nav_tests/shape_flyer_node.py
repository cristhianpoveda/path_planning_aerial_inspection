import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Empty, String, Float64MultiArray
from geometry_msgs.msg import Vector3  # Assuming speed_vector uses Vector3
from typing import List, Dict, Any
import csv
import os
from datetime import datetime

class ShapeFlyerNode(Node):
    def __init__(self) -> None:
        super().__init__('shape_flyer_node')
        
        # --- Publishers ---
        self.pub_enable = self.create_publisher(Empty, '/drone_1/command/enable_virtual_stick', 10)
        self.pub_takeoff = self.create_publisher(Empty, '/drone_1/command/takeoff', 10)
        self.pub_land = self.create_publisher(Empty, '/drone_1/command/land', 10)
        self.pub_disable = self.create_publisher(Empty, '/drone_1/command/abort_mission', 10)
        self.pub_stick = self.create_publisher(Float64MultiArray, '/drone_1/command/stick', 10)

        # --- Subscribers ---
        self.sub_trigger = self.create_subscription(String, '/drone_1/trigger_shape', self.trigger_callback, 10)
        self.sub_speed = self.create_subscription(Vector3, '/drone_1/speed_vector', self.speed_callback, 10)

        # --- Timers (20Hz = 0.05s) ---
        self.timer_period: float = 0.05 
        self.control_timer = self.create_timer(self.timer_period, self.control_loop)

        # --- FSM State Variables ---
        self.state: str = 'IDLE'  # States: IDLE, INIT, TAKEOFF, FLYING, HOVERING, LANDING
        self.state_start_time: float = 0.0
        
        # --- Flight Logic ---
        self.current_shape: List[Dict[str, float]] = []
        self.step_index: int = 0
        self.step_start_time: float = 0.0
        self.current_sticks: List[float] = [0.0, 0.0, 0.0, 0.0] # [yaw, throttle, roll, pitch]
        
        # --- Logging Data ---
        self.actual_speed: List[float] = [0.0, 0.0, 0.0] # [x, y, z]
        self.log_data: List[List[Any]] = []

        # --- Defined Shapes ---
        self.shapes: Dict[str, List[Dict[str, float]]] = {
            "cube": [
                {"duration": 0.5, "sticks": [0.0, 0.05, 0.0, 0.0]},  # Ascend
                {"duration": 0.5, "sticks": [0.0, 0.0, 0.0, 0.05]},  # Forward
                {"duration": 0.5, "sticks": [0.0, 0.0, 0.05, 0.0]},  # Right
                {"duration": 0.5, "sticks": [0.0, 0.0, 0.0, -0.05]}, # Backward
                {"duration": 0.5, "sticks": [0.0, 0.0, -0.05, 0.0]}, # Left
                {"duration": 0.5, "sticks": [0.0, -0.05, 0.0, 0.0]}, # Descend back to start
            ]
        }
        self.get_logger().info("Non-blocking Shape Flyer ready. Waiting for /drone_1/trigger_shape...")

    def speed_callback(self, msg: Vector3) -> None:
        """Continuously updates the drone's actual velocity."""
        self.actual_speed = [msg.x, msg.y, msg.z]

    def trigger_callback(self, msg: String) -> None:
        shape_name = msg.data.lower()
        if self.state != 'IDLE':
            self.get_logger().warn("Currently active! Ignoring new shape trigger.")
            return
            
        if shape_name in self.shapes:
            self.current_shape = self.shapes[shape_name]
            self.log_data.clear() # Reset log for new flight
            self.transition_state('INIT')
        else:
            self.get_logger().error(f"Unknown shape '{shape_name}'")

    def transition_state(self, new_state: str) -> None:
        """Handles the logic and timestamps for moving between FSM states."""
        self.state = new_state
        self.state_start_time = self.get_clock().now().nanoseconds / 1e9
        
        if new_state == 'INIT':
            self.get_logger().info("Enabling Virtual Stick...")
            self.pub_enable.publish(Empty())
        elif new_state == 'TAKEOFF':
            self.get_logger().info("Taking off... waiting for stabilization.")
            self.pub_takeoff.publish(Empty())
        elif new_state == 'FLYING':
            self.step_index = 0
            self.step_start_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info("Commencing shape execution.")
        elif new_state == 'HOVERING':
            self.get_logger().info("Shape complete. Hovering before land...")
            self.current_sticks = [0.0, 0.0, 0.0, 0.0]
        elif new_state == 'LANDING':
            self.get_logger().info("Landing...")
            self.pub_land.publish(Empty())
        elif new_state == 'IDLE':
            self.get_logger().info("Disabling Virtual Stick and saving logs...")
            self.pub_disable.publish(Empty())
            self.save_log()

    def control_loop(self) -> None:
        """20Hz continuous loop. Handles timeouts and state transitions."""
        if self.state == 'IDLE':
            return

        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed_state = current_time - self.state_start_time

        # --- State Machine Transitions ---
        if self.state == 'INIT' and elapsed_state > 0.5:
            self.transition_state('TAKEOFF')
            
        elif self.state == 'TAKEOFF' and elapsed_state > 5.0:
            self.transition_state('FLYING')
            
        elif self.state == 'FLYING':
            current_step = self.current_shape[self.step_index]
            elapsed_step = current_time - self.step_start_time

            if elapsed_step >= current_step["duration"]:
                self.step_index += 1
                if self.step_index >= len(self.current_shape):
                    self.transition_state('HOVERING')
                else:
                    self.step_start_time = current_time
                    self.get_logger().info(f"Step {self.step_index + 1}/{len(self.current_shape)}")
            else:
                self.current_sticks = current_step["sticks"]
                
        elif self.state == 'HOVERING' and elapsed_state > 2.0:
            self.transition_state('LANDING')
            
        elif self.state == 'LANDING' and elapsed_state > 4.0:
            self.transition_state('IDLE')

        # --- Publish 20Hz Heartbeat & Log ---
        if self.state in ['FLYING', 'HOVERING']:
            msg = Float64MultiArray()
            msg.data = self.current_sticks
            self.pub_stick.publish(msg)
            
            # Log the data for latency analysis (Timestamp, Commands, Actual Speeds)
            self.log_data.append([
                current_time, 
                *self.current_sticks, 
                *self.actual_speed
            ])

    def save_log(self) -> None:
        """Dumps the recorded flight telemetry into a CSV file."""
        if not self.log_data:
            return
            
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flight_latency_log_{timestamp_str}.csv"
        
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time(s)', 'Cmd_Yaw', 'Cmd_Throt', 'Cmd_Roll', 'Cmd_Pitch', 'Speed_X', 'Speed_Y', 'Speed_Z'])
                writer.writerows(self.log_data)
            self.get_logger().info(f"SUCCESS: Log saved to {os.path.abspath(filename)}")
        except Exception as e:
            self.get_logger().error(f"Failed to save log: {e}")

    def safe_shutdown(self) -> None:
        """Called automatically if the node crashes or Ctrl+C is pressed."""
        self.get_logger().warn("EMERGENCY SHUTDOWN TRIGGERED! Sending stop commands...")
        self.state = 'IDLE' # Break the FSM
        self.pub_stick.publish(Float64MultiArray(data=[0.0, 0.0, 0.0, 0.0]))
        self.pub_land.publish(Empty())
        self.pub_disable.publish(Empty())
        self.save_log() # Attempt to save whatever data we got before crash

def main(args=None):
    rclpy.init(args=args)
    node = ShapeFlyerNode()
    
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.safe_shutdown()
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()