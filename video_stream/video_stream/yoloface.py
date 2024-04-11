import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import sensor_msgs.msg as sensor_msgs
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
from ultralytics import YOLO
import torch
import sys


class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__('face_detection')

        qos_profile = QoSProfile(depth=10)
        self.image_sub = self.create_subscription(sensor_msgs.Image, '/image_raw', self.detect_face_callback, qos_profile=qos_profile)
        self.bridge = CvBridge()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.facemodel_path="/home/server/ros2_ws/src/video_stream/video_stream/yolov8m-face.pt"
        self.facemodel=YOLO(self.facemodel_path).to(self.device)
        print( self.device)


    def detect_face_callback(self, image):   
        frame = self.bridge.imgmsg_to_cv2(image, "bgr8")
        faces = self.detect_and_crop_face(frame)
        
        cv2.imshow('Detected Faces', frame)
        cv2.waitKey(1)


    def detect_and_crop_face(self, image):    
        detections= self.facemodel.predict(image,conf=0.7)
        faces = []
        cropped_faces = []
        for detection in detections:
            bboxes = detection.boxes
            for bbox in bboxes:
                x1,y1,x2,y2 = bbox.xyxy[0]
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                face_roi = image[y2:y1,x2:x1]
                faces.append((x1,y1,x2,y2))
                cropped_faces.append(face_roi)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return faces, cropped_faces
   
def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()