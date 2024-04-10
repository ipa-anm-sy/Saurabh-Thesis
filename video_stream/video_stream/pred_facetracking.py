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
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker

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
        self.tracker_path="/home/server/ros2_ws/src/deep_sort/deep/checkpoint/ckpt.t7"
        self.tracker= DeepSort(model_path=self.tracker_path,max_age=30)
        self.class_labels = ['Surprise','Fear', 'Disgust','Happy', 'Sad', 'Angry', 'Neutral']  
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = "/home/server/ros2_ws/src/video_stream/video_stream/rafdb_8830.pth"
        self.model = DDAMNet(num_class=7,num_head=2,pretrained=False)
        self.checkpoint= torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.frame_count = 0
        self.start_time = time.time()


    def detect_face_callback(self, image):   
        frame = self.bridge.imgmsg_to_cv2(image, "bgr8")
        self.frame_count += 1
        faces,cropped_faces = self.detect_and_crop_face(frame)
        for i, (startX, startY, endX, endY) in enumerate(faces):
            face_img= cropped_faces[i]
            label = self.get_expression_label(face_img)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)    
            cv2.putText(frame, label, (startX, startY - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        fps_text = f"Model FPS: {self.update_fps()}"
        frame=cv2.resize(frame,(1280,720),1)
        cv2.putText(frame, fps_text, (frame.shape[1] - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow('Detected Faces', frame)   
        cv2.waitKey(1)


    def detect_and_crop_face(self, image):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (1920,1080)), 1.0, (300, 300), (104.0, 177.0, 123.0)) # image, resizing pixel dimentions, scaling factor, blob size, normalization parameters.
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []
        cropped_faces = []
        if detections.any():   
            bboxes_xyxy=[]
            confidences=[]
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.8:  
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    bboxes_xyxy.append([startX, startY, endX, endY])
                    confidences.append(confidence)
                    bboxes_xywh=[]
                    for bbox in bboxes_xyxy:
                        x1,y1,x2,y2= bbox
                        w= x2-x1
                        h= y2-y1
                        bbox_xywh=[x1,y1,w,h]
                        bboxes_xywh.append(bbox_xywh)
                    bboxes_xywh=np.array(bboxes_xywh)
                    self.tracker.update(bboxes_xywh, confidences, image)
                    cropped_faces=[]
                    faces=[]
                    for track in self.tracker.tracker.tracks:
                        track_id=track.track_id
                        hits= track.hits
                        x1,y1,x2,y2=track.to_tlbr()
                        w= x2-x1
                        h= y2-y1
                        shift_percent=0.50
                        y_shift=int(h*shift_percent)
                        x_shift=int(w*shift_percent)
                        y1 += y_shift
                        y2 += y_shift
                        x1 += x_shift
                        x2 += x_shift
                        bbox_xywh=(x1,y1,w,h)
                        cv2.putText(image, f'id: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        face_roi = image[int(y1):int(y2), int(x1):int(x2)]
                        faces.append((int(x1), int(y1), int(x1+w), int(y1+h)))
                        cropped_faces.append(face_roi)      
        return faces, cropped_faces
    

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
        _, predicted = torch.max(output, 1)
        expression_label = self.class_labels[predicted.item()]
        return expression_label
           

def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()