import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import sensor_msgs.msg as sensor_msgs
import cv2
from cv_bridge import CvBridge

class video_stream(Node):
    def __init__(self):
        super().__init__('video_stream')
        self.br= CvBridge()
        qos_profile=QoSProfile(depth=10)
        self.image_sub = self.create_subscription(sensor_msgs.Image,"/image_raw",self.video_stream_callback,qos_profile= qos_profile)
        
    
    def video_stream_callback(self,image):
        
        cv_image = self.br.imgmsg_to_cv2(image, "bgr8")
        cv2.imshow("video", cv_image)
        cv2.waitKey(1)
    
def main(args=None):
        rclpy.init(args=args)
        node = video_stream()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
        main()