import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import sensor_msgs.msg as sensor_msgs
import cv2
from cv_bridge import CvBridge
import numpy as np

class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__('face_detection')
        qos_profile = QoSProfile(depth=10)
        self.image_sub = self.create_subscription(sensor_msgs.Image, '/image_raw', self.detect_face_callback, qos_profile= qos_profile)
        self.bridge = CvBridge()

    def detect_face_callback(self, image):
        try:

            frame = self.bridge.imgmsg_to_cv2(image,"bgr8")

        except Exception as e:

            self.get_logger().error(f"error converting image:{e}")
            return
        
        face = self.detect_and_crop_face(frame)

        if face is not None:

            x, y, w, h = self.face_coordinates
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow('Detected Face', frame)

            cv2.waitKey(1)

    def detect_and_crop_face(self, image):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)

        if len(faces) == 0:
            return None
        
        x, y, w, h = faces[0]
        self.face_coordinates = (x, y, w, h)
        face_roi = image[y:y+h, x:x+w]


        return face_roi

def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()