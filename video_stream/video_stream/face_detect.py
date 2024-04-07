import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import sensor_msgs.msg as sensor_msgs
import cv2
from cv_bridge import CvBridge
import numpy as np
import os


class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__('face_detection')
        qos_profile = QoSProfile(depth=10)
        self.image_sub = self.create_subscription(sensor_msgs.Image, '/image_raw', self.detect_face_callback, qos_profile= qos_profile)
        self.bridge = CvBridge()
        model_dir= "/home/server/ros2_ws/src/video_stream/video_stream"
        self.caffemodel = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")#
        self.prototxt = os.path.join(model_dir, "deploy.prototxt.txt")
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.caffemodel)


    def detect_face_callback(self, image):   
        frame = self.bridge.imgmsg_to_cv2(image, "bgr8")
        faces = self.detect_and_crop_face(frame)
        cv2.imshow('Detected Faces', frame)
        cv2.waitKey(1)
        
        
    def detect_and_crop_face(self, image):    
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []
        cropped_faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:  
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face_roi = image[startY:endY, startX:endX]
                faces.append((startX, startY, endX, endY))
                cropped_faces.append(face_roi)
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        return faces, cropped_faces


        return face_roi

def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()