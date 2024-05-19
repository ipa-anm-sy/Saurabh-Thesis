import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
from ultralytics import YOLO
import time
import sys
from ultralytics.utils.plotting import Annotator, colors
sys.path.append("/home/server/ros2_ws/src/DDAMFN/DDAMFN++")
from networks.DDAM import DDAMNet

class FaceDetectionNode:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.facemodel_path = "/home/server/ros2_ws/src/video_stream/video_stream/yolov8m-face.pt"
        self.facemodel = YOLO(self.facemodel_path).to(self.device)
        self.class_labels = ['Surprise','Fear','Disgust','Happy','Sad','Angry','Neutral']  # class with my trained models
        #self.class_labels = ['Neutral','Happy','Sad','Surprise','Fear','Disgust','Angry']
        self.model_path = "/home/server/ros2_ws/src/video_stream/video_stream/rafdb_8980.pth"
        self.model = DDAMNet(num_class=7, num_head=2, pretrained=False)
        self.checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.track_history = defaultdict(list)
        self.frame_count = 0
        self.frame_number = 0
        self.start_time = time.time()
        self.video_path = "/home/server/ros2_ws/src/video_stream/videos/cam_3.1.mp4"  # Update with the path to your video file
        self.names=self.facemodel.names
        
    def detect_faces(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open the video file.")
            return
        result=cv2.VideoWriter('cam_3.1.mp4',cv2.VideoWriter_fourcc(*"mp4v"),1.0,(1280,720))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame=cv2.resize(frame,(1280,720),1)
            self.frame_count += 1
            self.frame_number += 1
            frames,fps_text = self.detect_and_crop_face(frame)
            
            for frame in frames:
                cv2.putText(frame, fps_text, (frame.shape[1] - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                
                result.write(frame)
            cv2.imshow('Detected Faces', frame)        
            cv2.waitKey(1) 
            #result=cv2.VideoWriter('expression.mp4',cv2.VideoWriter_fourcc(*"mp4v"),20.0,(1920,1080))    
             
        cap.release()
        result.release()
        print("video has been saved") 
        cv2.destroyAllWindows()

    def detect_and_crop_face(self, image):
        
        faces=[]
        cropped_faces=[]
        results = self.facemodel.track(image, persist=True,conf=0.7)#<---------------------------------------------------------------------------
        frames=[]
        track_ids=[]
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            self.annotator = Annotator(image, line_width=3, example=str(self.names))
 
            for box, track_id, cls in zip(boxes, track_ids, clss):
                x1=int(box[0])
                y1=int(box[1])
                x2=int(box[2])
                y2=int(box[3])
                
                face_roi = image[int(y1):int(y2), int(x1):int(x2)]
                faces.append((int(x1), int(y1), int(x2), int(y2)))
                cropped_faces.append(face_roi)
                
                label = self.get_expression_label(face_roi)
                cv2.putText(image, label, (x1, y1- 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                 
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center
                cv2.putText(image, f'id: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                track = self.track_history[track_id]  # Tracking Lines plot
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 300:
                    track.pop(0)
                
                
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                
                frames.append(image)
                f=open("emotions.txt","a+")
                f.write(str(track_id)+","+str(label)+ ","+str(self.frame_number) +"\n")
                f.close()
        fps_text = f"Model FPS: {self.update_fps()}"
        cv2.putText(image, fps_text, (image.shape[1] - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
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
        
        if confidence> 0.8:
        #_, predicted = torch.max(output, 1)
            expression_label = self.class_labels[predicted.item()]
        else :
           expression_label= " "
        return expression_label
def main():
    
    node = FaceDetectionNode()
    node.detect_faces()

if __name__ == '__main__':
    main()