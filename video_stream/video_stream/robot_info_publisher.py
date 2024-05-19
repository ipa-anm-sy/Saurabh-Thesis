import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class ROBINFOSUB(Node):
    def __init__(self):
        super().__init__('ROBINFOSUB')
        
        
        self.vel_sub = self.create_subscription(Twist, '/cmd_vel', self.robot_info, 10)
    
        self.rob_state_pub = self.create_publisher(String, '/current_state', 10)
        
        self.linear_x = 0.0
        self.linear_y = 0.0
        self.angular_z = 0.0
        
        self.robot_state = String()
        self.robot_state.data = "The robot is Stationary"
        
        self.rob_state_pub.publish(self.robot_state)
        
        self.last_cmd_vel_time = self.get_clock().now()
        
        self.timer = self.create_timer(0.1, self.check_stationary_state)
        
    def robot_info(self, msg):
        self.last_cmd_vel_time = self.get_clock().now()
        
        self.linear_x = msg.linear.x
        self.linear_y = msg.linear.y
        self.angular_z = msg.angular.z
        
        if self.linear_x > 0 and self.angular_z == 0:
            self.robot_state.data = "The robot is going forward"
        elif self.linear_y > 0 and self.angular_z == 0:  
            self.robot_state.data = "The robot is going backward"
        elif self.linear_x > 0 and self.angular_z > 0:
            self.robot_state.data = "The robot is going Forward and Turning Right"
        elif self.linear_x > 0 and self.angular_z < 0:
            self.robot_state.data = "The robot is going Forward and Turning Left"
        elif self.linear_x == 0 and self.angular_z < 0: 
            self.robot_state.data = "The robot is Turning Left"
        elif self.linear_x == 0 and self.angular_z > 0: 
            self.robot_state.data = "The robot is Turning Right"
        elif self.linear_x == 0 and self.angular_z == 0 and self.linear_y == 0:
            self.robot_state.data = "The robot is Stationary"
        
        self.rob_state_pub.publish(self.robot_state)
        
    def check_stationary_state(self):
        current_time = self.get_clock().now()
        
        if (current_time - self.last_cmd_vel_time).nanoseconds > 1e9:
            self.robot_state.data = "The robot is Stationary"
            self.rob_state_pub.publish(self.robot_state)
        

def main(args=None):
    rclpy.init(args=args)
    node = ROBINFOSUB()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()