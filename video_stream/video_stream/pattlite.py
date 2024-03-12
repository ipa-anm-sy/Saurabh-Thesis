
import os
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import sensor_msgs.msg as sensor_msgs
import cv2
from cv_bridge import CvBridge
import numpy as np
# import torch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from keras.models import Model, load_model
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from PIL import Image


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


    def detect_face_callback(self, image):
        try:
            frame = self.bridge.imgmsg_to_cv2(image, "bgr8")
        except Exception as e:
            self.get_logger().error(f"error converting image:{e}")
            return
        
        faces,cropped_faces = self.detect_and_crop_face(frame)

        for (startX, startY, endX, endY) in faces:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        for i in cropped_faces:
            
            label = self.get_expression_label(i)
            cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
    
    def get_expression_label(self, image):
        class_labels =  ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

        if tf.test.is_gpu_available():
            device = '/GPU:0'  # Use GPU
        else:
            device = '/CPU:0'  # Use CPU

       
        with tf.device(device):
            image= Image.fromarray(image)
            image = tf.image.resize(image, (224, 224))
            img_array = keras.preprocessing.image.img_to_array(image)
            img_array_expanded_dims = np.expand_dims(img_array, axis=0)
            img_array_expanded_dims_copy = np.copy(img_array_expanded_dims)
            input= keras.applications.mobilenet.preprocess_input(img_array_expanded_dims_copy)

            model_path="/home/server/ros2_ws/src/video_stream/video_stream/rafdb.h5"
            patt = tf.keras.models.load_model(model_path, compile=False)
            
            with tf.GradientTape() as tape:
                output=patt(input)

            predicted = tf.argmax(output, axis=1).numpy()[0]
            print('predicted is:',predicted)
            expression_label = class_labels[predicted] 
            

            return expression_label

def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
