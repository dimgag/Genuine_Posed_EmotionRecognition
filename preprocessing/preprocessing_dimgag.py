import os 
import json
import cv2

FPS = 20

def video2frames(output_path, video, filename):
    print(video)
    cap = cv2.VideoCapture(video)

    # Get FPS from the video
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the number of frames to skip in order to have the desired amount
    skip_frames = video_fps // FPS


    curr_frame, i = 0, 0
    while cap.isOpened():
        curr_frame += 1
        ret, frame = cap.read()
        if ret == False:
            break
        if curr_frame % skip_frames == 0:
            i += 1
            cv2.imwrite(os.path.join(output_path, filename + "{:02d}.jpg".format(i)), frame)
    cap.release()



def get_file_path(data_dir):
    labels = json.load(open("/Users/dim__gag/git/Genuine_Posed_EmotionRecognition/data/emotion_labels.txt"))
    print(labels)
    dirlist = os.listdir(data_dir)
    
    for folder in dirlist:
        subject_path = os.path.join(data_dir, folder)
        print(subject_path)

        for vid in os.listdir(subject_path):
            filename = vid[0:2]
            output_path = os.path.join(subject_path, labels[vid])
            print(output_path)

            if not os.path.exists(output_path):
                os.makedirs(output_path)
            video2frames(output_path, os.path.join(subject_path, vid), filename)
        break
    return ""




data_dir = "data"
get_file_path(data_dir)




