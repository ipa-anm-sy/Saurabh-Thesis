

import os
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import sensor_msgs.msg as sensor_msgs
import cv2
from cv_bridge import CvBridge
import numpy as np
import torch

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import time
import sys
sys.path.append("/home/server/ros2_ws/src")
from DDAMFN.networks.DDAM import DDAMNet





class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__('face_detection')

        qos_profile = QoSProfile(depth=10)
        self.image_sub = self.create_subscription(sensor_msgs.Image, ' ', self.detect_face_callback, qos_profile=qos_profile)
       
        self.bridge = CvBridge()

        model_dir= "/home/server/ros2_ws/src/video_stream/video_stream"
        self.caffemodel = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        self.prototxt = os.path.join(model_dir, "deploy.prototxt.txt")
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.caffemodel)

        self.frame_count = 0
        self.start_time = time.time()
        self.frames=[]
        
        
    def detect_face_callback(self, image):
        
        frame = self.bridge.imgmsg_to_cv2(image, "bgr8")
        
        
        self.frame_count += 1
        self.frames.append(frame)
        
        if len(self.frames)==16:

            #------------------------------------------------------------- Detecting and cropped faces----------------------------------------------------------
            faces,cropped_faces = self.detect_and_crop_face(self.frames) 

            #------------------------------------------------------------- Predicting expression----------------------------------------------------------------
            labels = self.get_expression_label(cropped_faces)
            #------------------------------------------------------------- output with labels ------------------------------------------------------------------
            for frame, face_locations, label_list in zip(self.frames, faces, labels):
                
                for (startX, startY, endX, endY), label in zip(face_locations, label_list):
                    
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)  
                    
                    cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                fps =  self.update_fps()
                fps_text = f"Model FPS: {fps}"
                if fps!=0.0:
                    cv2.putText(frame, fps_text, (frame.shape[1] - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow('Detected Faces', frame)
                
                cv2.waitKey(delay=2)
            
            self.frames =[]



    def detect_and_crop_face(self, images):
        face_list=[]
        cropped_faces_list=[] 
        
        for image in images:
            image_np = np.array(image)
            (h, w) = image_np.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

            self.net.setInput(blob)
            detections = self.net.forward()

            faces = []
            cropped_faces = []
            
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.7:  
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    face_roi = image_np[startY:endY, startX:endX]
                    faces.append((startX, startY, endX, endY))
                    cropped_faces.append(face_roi)
                    cv2.rectangle(image_np, (startX, startY), (endX, endY), (0, 255, 0), 2)

            face_list.append(faces)
            cropped_faces_list.append(cropped_faces)
        return face_list, cropped_faces_list
    

    def update_fps(self):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        fps = self.frame_count / elapsed_time
        self.start_time =  end_time
        self.frame_count = 0
        return round(fps, 2)
    
    def get_expression_label(self, images):
        
        class_labels = ['Surprise','Fear', 'Disgust','Happy', 'Sad', 'Angry', 'Neutral']  
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_path = "/home/server/ros2_ws/src/video_stream/video_stream/rafdb.pth"
        model = DDAMNet(num_class=7,num_head=2,pretrained=False) # num_head is attention heads, it can be trained with changing the number to 3,4 or something else.
        checkpoint= torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

           
        preprocess = transforms.Compose([
                transforms.Resize((112,112)), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        expression_labels = []
        for image_list in images:
            expression_labels_list = []
            for image in image_list:
                face_img = Image.fromarray(image)
                face_img = preprocess(face_img)
                face_img = face_img.unsqueeze(0)  
                face_img = face_img.to(device)
                            
                with torch.no_grad():
                    output = model(face_img)

                
                if isinstance(output, tuple):
                
                    output = output[0] 
                    

                _, predicted = torch.max(output, 1)

                

                expression_label = class_labels[predicted.item()]
                expression_labels_list.append(expression_label)
            expression_labels.append(expression_labels_list)   
        return expression_labels
        

        

def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
