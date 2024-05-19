import pandas as pd
import matplotlib.pyplot as plt


face1=[[],[]]
face2=[[],[]]
face3=[[],[]]
face4=[[],[]]
face5=[[],[]]
face6=[[],[]]
face7=[[],[]]
face8=[[],[]]
face9=[[],[]]
face10=[[],[]]


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
    elif file["Face_id"][i]==3:
        face3[0].append(file["Frame number"][i])
        face3[1].append(file["Emotion"][i])
    elif file["Face_id"][i]==4:
        face4[0].append(file["Frame number"][i])
        face4[1].append(file["Emotion"][i])
    elif file["Face_id"][i]==5:
        face5[0].append(file["Frame number"][i])
        face5[1].append(file["Emotion"][i])
    elif file["Face_id"][i]==5:
        face6[0].append(file["Frame number"][i])
        face6[1].append(file["Emotion"][i])
    elif file["Face_id"][i]==5:
        face7[0].append(file["Frame number"][i])
        face7[1].append(file["Emotion"][i])
    elif file["Face_id"][i]==5:
        face8[0].append(file["Frame number"][i])
        face8[1].append(file["Emotion"][i])
    elif file["Face_id"][i]==5:
        face9[0].append(file["Frame number"][i])
        face9[1].append(file["Emotion"][i])
    elif file["Face_id"][i]==5:
        face10[0].append(file["Frame number"][i])
        face10[1].append(file["Emotion"][i])
        
plt.plot(face1[0],face1[1])
plt.yticks(range(len(Ylabel)), Ylabel)
plt.xlabel('Frame Number')  
plt.ylabel('Emotions')
plt.title("graph 1")
plt.show()

plt.plot(face2[0],face2[1])
plt.yticks(range(len(Ylabel)), Ylabel)
plt.xlabel('Frame Number')  
plt.ylabel('Emotions')
plt.title("Graph 2")
plt.show()

plt.plot(face3[0],face3[1])
plt.yticks(range(len(Ylabel)), Ylabel)
plt.xlabel('Frame Number')  
plt.ylabel('Emotions')
plt.title("Graph 3")
plt.show()

plt.plot(face4[0],face4[1])
plt.yticks(range(len(Ylabel)), Ylabel)
plt.xlabel('Frame Number')  
plt.ylabel('Emotions')
plt.title("Graph 4")
plt.show()

plt.plot(face5[0],face5[1])
plt.yticks(range(len(Ylabel)), Ylabel)
plt.xlabel('Frame Number')  
plt.ylabel('Emotions')
plt.title("Graph 5")
plt.show()

plt.plot(face6[0],face6[1])
plt.yticks(range(len(Ylabel)), Ylabel)
plt.xlabel('Frame Number')  
plt.ylabel('Emotions')
plt.title("Graph 6")
plt.show()

plt.plot(face7[0],face7[1])
plt.yticks(range(len(Ylabel)), Ylabel)
plt.xlabel('Frame Number')  
plt.ylabel('Emotions')
plt.title("Graph 7")
plt.show()

plt.plot(face8[0],face8[1])
plt.yticks(range(len(Ylabel)), Ylabel)
plt.xlabel('Frame Number')  
plt.ylabel('Emotions')
plt.title("Graph 8")
plt.show()

plt.plot(face9[0],face9[1])
plt.yticks(range(len(Ylabel)), Ylabel)
plt.xlabel('Frame Number')  
plt.ylabel('Emotions')
plt.title("Graph 9")
plt.show()

plt.plot(face10[0],face10[1])
plt.yticks(range(len(Ylabel)), Ylabel)
plt.xlabel('Frame Number')  
plt.ylabel('Emotions')
plt.title("Graph 10")
plt.show()