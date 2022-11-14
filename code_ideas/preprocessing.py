import os
import json
import cv2
import re
# Number of fps per minute that we want to take from a video
DESIRED_FPS = 20

"""
takes frames from a video and save them in the appropriate folder
"""
def video_to_frames(output_path,vid,filename):
    print(vid)
    cap = cv2.VideoCapture(vid)

    #get fps from the video
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    #calculates the number of frames to skip in order to have the desired amount
    skip_frames= video_fps//DESIRED_FPS

    curr_frame,i = 0 , 0    
    while(cap.isOpened()):
        curr_frame += 1
        ret, frame = cap.read()
        if ret == False: 
            break
        if curr_frame % skip_frames==0:
            i +=1
            # cv2.imwrite("frame{}.jpg".format(i), frame)
            cv2.imwrite(os.path.join(output_path, filename +"{:02d}.jpg".format(i)), frame)
    cap.release()
# def video_to_frames(vid):
#     print(vid)


"""
given a path loop through the subject and call the function to save the frames of a video
"""
def get_file_path(DATA_DIR):
    label = json.load(open("label.txt"))
    print(label)
    dirlist = os.listdir(DATA_DIR)
    for folder in dirlist:
        subject_path = os.path.join(DATA_DIR, folder)
        print(subject_path)
        for vid in os.listdir(subject_path):
            filename= vid[0:2]
            output_path = os.path.join(subject_path,label[vid])
            print(output_path)
            if not os.path.exists(output_path):
                os.makedirs(output_path)    
            video_to_frames(output_path,os.path.join(subject_path,vid), filename)
        break
        # [lambda vid: video_to_frames(os.path.join(subject_path,vid)) for vid in os.listdir(subject_path)]
        # videos = os.listdir(DATA_DIR+"\\"+folder)
        # print(videos)
        
    return ""





DATA_DIR= r"C:\Users\luca9\Desktop\Thesis\Code\Datasets\SASE-FE database\FakeTrue_DB"
get_file_path(DATA_DIR)
# video_to_frames(r"C:\Users\luca9\Desktop\Thesis\Code\Datasets\SASE-FE database\FakeTrue_DB\age\D2N2Sur.MP4")
