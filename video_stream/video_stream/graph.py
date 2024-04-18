import pandas as pd
import matplotlib.pyplot as plt
face1=[[],[]]
face2=[[],[]]
Ylabel=['Neutral','Happy','Sad','Surprise','Fear','Disgust','Angry']


file=pd.read_csv("/home/server/ros2_ws/src/video_stream/video_stream/emotions.csv")
file.columns=["Face_id","Emotion","Frame number"]
for i in range(len(file)):
    
    if file["Face_id"][i]==1:
        face1[0].append(file["Frame number"][i])
        face1[1].append(file["Emotion"][i])
    elif file["Face_id"][i]==2:
        face2[0].append(file["Frame number"][i])
        face2[1].append(file["Emotion"][i])

plt.plot(face1[0],face1[1])
plt.yticks(range(len(Ylabel)), Ylabel)
plt.title("graph 1")
plt.show()

plt.plot(face2[0],face2[1])
plt.yticks(range(len(Ylabel)), Ylabel)
plt.title("Graph 2")
plt.show()