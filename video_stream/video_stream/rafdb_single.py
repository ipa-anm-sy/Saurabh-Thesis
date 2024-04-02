

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
from PIL import Image
import time
import sys
sys.path.append("/home/server/ros2_ws/src")
from DDAMFN.networks.DDAM import DDAMNet





class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__('face_detection')

        qos_profile = QoSProfile(depth=10)
        self.image_sub = self.create_subscription(sensor_msgs.Image, '/image_raw', self.detect_face_callback, qos_profile=qos_profile)
       
        self.bridge = CvBridge()

        model_dir= "/home/server/ros2_ws/src/video_stream/video_stream"
        self.caffemodel = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        self.prototxt = os.path.join(model_dir, "deploy.prototxt.txt")
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.caffemodel)

        self.frame_count = 0
        self.start_time = time.time()

    def detect_face_callback(self, image):
        try:
            frame = self.bridge.imgmsg_to_cv2(image, "bgr8")
        except Exception as e:
            self.get_logger().error(f"error converting image:{e}")
            return
        
        self.frame_count += 1
        
        faces,cropped_faces = self.detect_and_crop_face(frame)

        for i, (startX, startY, endX, endY) in enumerate(faces):
            face_img= cropped_faces[i]
            label = self.get_expression_label(face_img)
            
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)  
            
            cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        fps_text = f"Model FPS: {self.update_fps()}"
        cv2.putText(frame, fps_text, (frame.shape[1] - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

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
    

    def update_fps(self):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        fps = self.frame_count / elapsed_time
        self.start_time =  end_time
        self.frame_count = 0
        return round(fps, 2)
    
    def get_expression_label(self, image):
        
        class_labels = ['Surprise','Fear', 'Disgust','Happy', 'Sad', 'Angry', 'Neutral']  
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_path = "/home/server/ros2_ws/src/video_stream/video_stream/rafdb.pth"
        model = DDAMNet(num_class=7,num_head=2,pretrained=False)
        checkpoint= torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

           
        preprocess = transforms.Compose([
                transforms.Resize((112,112)), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        
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

        return expression_label
        

        

def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()