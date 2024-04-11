import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import sensor_msgs.msg as sensor_msgs
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import sys
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors



    

class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__('face_detection')

        qos_profile = QoSProfile(depth=10)
        self.image_sub = self.create_subscription(sensor_msgs.Image, '/image_raw', self.detect_face_callback, qos_profile=qos_profile)
        self.bridge = CvBridge()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.facemodel_path="/home/server/ros2_ws/src/video_stream/video_stream/yolov8m-face.pt"
        self.facemodel=YOLO(self.facemodel_path).to(self.device)
        self.class_labels = ['Surprise','Fear', 'Disgust','Happy', 'Sad', 'Angry', 'Neutral']  
        self.model_path = "/home/server/ros2_ws/src/video_stream/video_stream/rafdb_8830.pth"
        self.frame_count = 0
        self.start_time = time.time()
        self.track_history = defaultdict(list)
        self.names = self.facemodel.model.names

    def detect_face_callback(self, image):   
        frame = self.bridge.imgmsg_to_cv2(image, "bgr8")
        self.frame_count += 1
        #faces,cropped_faces = self.detect_and_crop_face(frame)
        results= self.detect_and_crop_face(frame)
        frame = self.bridge.imgmsg_to_cv2(image, "bgr8")
        self.frame_count += 1
        faces,cropped_faces = self.detect_and_crop_face(frame)
        for i, (startX, startY, endX, endY) in enumerate(faces):
            face_img= cropped_faces[i]
            #label = self.get_expression_label(face_img)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)    
            #cv2.putText(frame, label, (startX, startY - 30),
            #cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        fps_text = f"Model FPS: {self.update_fps()}"
        #frame=cv2.resize(frame,(1280,720),1)
        cv2.putText(frame, fps_text, (frame.shape[1] - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow('Detected Faces', frame)   
        cv2.waitKey(1)
       


    def detect_and_crop_face(self, image):
        faces=[]
        cropped_faces=[]
        results = self.facemodel.track(image, persist=True)
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            print("start\n")
            print(boxes)
            print("\nend")
            track_ids = results[0].boxes.id.int().cpu().tolist()
            print("track_ids",track_ids)
            annotator = Annotator(image, line_width=2,example=str(self.names))
            clss = results[0].boxes.cls.cpu().tolist()
            for box, track_id, cls in zip(boxes, track_ids, clss):
                x1=int(box[0])
                y1=int(box[1])
                x2=int(box[2])
                y2=int(box[3])
                
                face_roi = image[int(y1):int(y2), int(x1):int(x2)]
                faces.append((int(x1), int(y1), int(x2), int(y2)))
                cropped_faces.append(face_roi)
                #annotator.box_label(box, str(self.names[cls]), color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center
                cv2.putText(image, f'id: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                track = self.track_history[track_id]  # Tracking Lines plot
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 100:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                print("points", points)
                #cv2.polylines(image, [points], isClosed=False, color=colors(cls, True), thickness=2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                
                # Check if detection inside region
        return faces, cropped_faces
    

    def update_fps(self):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        fps = self.frame_count / elapsed_time
        self.start_time =  end_time
        self.frame_count = 0
        return round(fps, 2)
    

def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()