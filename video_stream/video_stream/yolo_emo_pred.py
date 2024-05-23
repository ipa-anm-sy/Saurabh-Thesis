import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from message_filters import ApproximateTimeSynchronizer, Subscriber
from PIL import Image
import time
from datetime import datetime
from pytz import timezone
import sys
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from collections import defaultdict
# sys.path.append("video_strea/video_stream/DDAMFN/DDAMFN++")
sys.path.append("/home/server/ros2_ws/src/DDAMFN/DDAMFN++")
from networks.DDAM import DDAMNet


class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__('face_detection')
        self.place = input("Enter place: ")
        self.temperature = input("Enter temperature: ")
        self.light_condition = input("Enter light condition: ")
        self.robot_reliability = input("Enter robot reliability: ")
        self.num_of_people = input("Enter number of people: ")
        self.task_number = input("Enter the Task number: ")
        self.write_exp_info()
        
        self.qos_profile = QoSProfile(depth=10)
        #self.image_sub = self.create_subscription(sensor_msgs.Image, '/image_raw', self.detect_face_callback, qos_profile=self.qos_profile)
       
        
        self.image_sub = Subscriber(self,sensor_msgs.Image, "/image_raw", qos_profile=self.qos_profile)
        self.msg_sub=Subscriber(self,std_msgs.String, "/current_state", qos_profile=10)
        self.ts=ApproximateTimeSynchronizer([self.image_sub,self.msg_sub],10,0.1,allow_headerless=True)
        self.ts.registerCallback(self.detect_face_callback)
        
        self.bridge = CvBridge()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.facemodel_path="/home/server/ros2_ws/src/video_stream/video_stream/yolov8m-face.pt"
        self.facemodel=YOLO(self.facemodel_path).to(self.device)
        self.class_labels = ['Surprise','Fear','Confused','Happy','Sad','Angry','Neutral']  # class with other trained models
        #self.class_labels = ['Neutral','Happy','Sad','Surprise','Fear','Disgust','Angry']   # class with pre trained model (rafdb_8617.pth)
        self.model_path = "/home/server/ros2_ws/src/video_stream/video_stream/rafdb_8980.pth"
        self.model = DDAMNet(num_class=7,num_head=2,pretrained=False)
        self.checkpoint= torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.names = self.facemodel.model.names
        self.track_history = defaultdict(list)
        self.frame_count = 0
        self.frame_number = 0
        self.start_time = time.time()
        self.vid_array=[]
        self.video_path = "/home/server/ros2_ws/src/video_stream/cam_vid.mp4"
        self.video = cv2.VideoWriter(self.video_path,cv2.VideoWriter_fourcc(*'mp4v'), 2, (1080,720))  
        
        
    def write_exp_info(self):
        file=open("/home/server/ros2_ws/src/video_stream/video_stream/emotions.txt","a+")
        fmt = "%Y-%m-%d %H:%M:%S %Z%z"
        zone = 'Europe/Berlin'
        now_time = datetime.now(timezone(zone))  
        
        file.write(str(now_time.strftime(fmt))+"\n")
        file.write(str("Task_number: ")+str(self.task_number)+"\n")
        file.write(str("robot_reliability: ")+str(self.robot_reliability)+"\n")
        file.write(str("place: ")+str(self.place)+"\n")
        file.write(str("temperature: ")+str(self.temperature)+"\n")
        file.write(str("lighting_condition: ")+str(self.light_condition)+"\n")
        file.write(str("num_of_people: ")+str(self.num_of_people)+"\n")

        file.close() 
        
            
    def detect_face_callback(self, image,msg): 
        msg=msg  
        frame = self.bridge.imgmsg_to_cv2(image, "bgr8") 
        self.frame_count += 1
        self.frame_number+=1
        frames,fps_text=self.detect_and_crop_face(frame,msg)
        for frame in (frames): 
            cv2.putText(frame, fps_text, (frame.shape[1] - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1) 
            frame=cv2.resize(frame,(1080,720),1)            
            cv2.imshow('Detected Faces', frame)
            self.video.write(frame)
            cv2.waitKey(1) 
        
               
    def detect_and_crop_face(self, image,msg):
        faces=[]
        cropped_faces=[]
        results = self.facemodel.track(image, persist=True)
        
        frames=[]
        track_ids=[]
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            conf = results[0].boxes.conf.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            for box, track_id, cls,conf in zip(boxes, track_ids, clss,conf):
                x1=int(box[0])
                y1=int(box[1])
                x2=int(box[2])
                y2=int(box[3])
                H=int (y2-y1)
                W=int (x2-x1)
                print(x1,y1,x2,y2)
                cv2.putText(image,f'H:{H}', (x1+100, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.putText(image,f'W:{W}', (x1+150, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                face_roi = image[int(y1):int(y2), int(x1):int(x2)]
                
                
                faces.append((int(x1), int(y1), int(x2), int(y2)))
                cropped_faces.append(face_roi)
                
                confidence=float(conf)
                cv2.putText(image,f'{confidence:.3f}', (x1+50, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                
                label = self.get_expression_label(face_roi)
                cv2.putText(image, label, (x1, y1- 30),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)
                 
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center
                cv2.putText(image, f'id: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                track = self.track_history[track_id]  
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 100:
                    track.pop(0)
               
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                frames.append(image)
                msg=msg.data
                f=open("/home/server/ros2_ws/src/video_stream/video_stream/emotions.txt","a+")
                f.write(str(track_id)+","+str(label)+ ","+str(self.frame_number)+","+str(msg)+"\n")
                f.close()
        fps_text = f"Model FPS: {self.update_fps()}"
             
        return frames,fps_text
    
    

    
        
    def update_fps(self):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        fps = self.frame_count / elapsed_time
        self.start_time =  end_time
        self.frame_count = 0
        return round(fps, 2)
    
    
    def get_expression_label(self, image):
        self.model.eval()   
        preprocess = transforms.Compose([
                     transforms.Resize((112,112)), 
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                     ])
        face_img = Image.fromarray(image)
        face_img = preprocess(face_img)
        face_img = face_img.unsqueeze(0)  
        face_img = face_img.to(self.device)                   
        with torch.no_grad():
            output = self.model(face_img)
        if isinstance(output, tuple):
            output = output[0]
            confidence_values = F.softmax(output,1)
        _, predicted  =torch.max(confidence_values,1) 
        confidence =torch.max(confidence_values,1) 
        confidence= confidence.values.item()
        
        if confidence> 0.7:
        #_, predicted = torch.max(output, 1)
            expression_label = self.class_labels[predicted.item()]
        else :
            expression_label= "no prediction"
        return expression_label
           
        
def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()