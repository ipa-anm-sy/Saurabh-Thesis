import os

def filter_emotions(file_path, output_file_path, min_consecutive=4):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    filtered_lines = []
    previous_emotions = {}
    buffer = []

    def buffer_filter():
        nonlocal buffer
        if buffer:
            if len(buffer) >= min_consecutive:
                filtered_lines.extend(buffer)
            buffer = []

    for line in lines:
        if any(keyword in line for keyword in ['Task_number', 'robot_reliability', 'place', 'temperature', 'lighting_condition', 'num_of_people']):
            buffer_filter()
            filtered_lines.append(line)
            continue
        
        parts = line.strip().split(',')
        if len(parts) == 4:
            face_id = parts[0]
            emotion = parts[1]
            frame_number = int(parts[2])
            current_state = parts[3]

            if face_id not in previous_emotions:
                previous_emotions[face_id] = {'last_emotion': None, 'count': 0}

            last_emotion_info = previous_emotions[face_id]
            
            if last_emotion_info['last_emotion'] == emotion:
                last_emotion_info['count'] += 1
            else:
                buffer_filter()
                last_emotion_info['last_emotion'] = emotion
                last_emotion_info['count'] = 1
            
            buffer.append(line)
    
    buffer_filter()
    
    with open(output_file_path, 'w') as output_file:
        output_file.writelines(filtered_lines)

if __name__ == "__main__":
    input_file_path = "/home/server/ros2_ws/src/video_stream/video_stream/emotions.txt"
    output_file_path = "/home/server/ros2_ws/src/video_stream/video_stream/filtered_emotions.txt"
    filter_emotions(input_file_path, output_file_path)
    print("Filtering complete. Check the filtered_emotions.txt file.")