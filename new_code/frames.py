import cv2
import os
import argparse
parser = argparse.ArgumentParser(description="Generate frames from videos")
parser.add_argument('-i', '--input_dir', type=str, default = None, help='input directory')
parser.add_argument('-o', '--output_dir', type=str, default = None, help='output_directory for the frames')
parser.add_argument('-f', '--frames', type=int, default = 5, help='Set the number of frames to be extracted from every video')
parser.add_argument('-s', '--size', type=int, default = 224, help='Set the size of frames')
args = parser.parse_args()

def generate_frames(input_dir, output_dir, num_frames, size):
    # Define the number of frames to extract
    num_frames = num_frames

    # Loop through each video in the input directory
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if os.path.isdir(folder_path):
            for video_file in os.listdir(folder_path):
                video_path = os.path.join(folder_path, video_file)
                # Open the video file
                video = cv2.VideoCapture(video_path)
                # Get the total number of frames
                num_frames_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                # Calculate the step size to extract 'num_frames' frames
                step_size = max(num_frames_total // num_frames, 1)
                # Initialize the frame count
                count = 0
                # Loop through each frame in the video
                while True:
                    # Read the next frame
                    ret, frame = video.read()
                    # If there are no more frames, break the loop
                    if not ret:
                        break
                    # If the frame count is a multiple of the step size, save the frame
                    if count % step_size == 0:
                        # Construct the output filename
                        output_folder = os.path.join(output_dir, folder, video_file.split('.')[0])
                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder)
                        output_filename = os.path.join(output_folder, f'frame_{count // step_size:04d}.jpg')
                        # Resize the frame to a standard size (optional)
                        frame = cv2.resize(frame, (size, size))
                        
                        # Crop the face from the frame (optional)
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
                        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        gray = cv2.GaussianBlur(frame, (25,25), 0)
                        face = face_cascade.detectMultiScale(gray, 1.1, 4)
                        for (x,y,w,h) in face:
                            frame = frame[y:y+h, x:x+w]
                        
                        # Save the frame to disk
                        cv2.imwrite(output_filename, frame)
                    # Increment the frame count
                    count += 1
                    # If we have extracted 'num_frames' frames, break the loop
                    if count // step_size >= num_frames:
                        break
                # Release the video file
                video.release()



def main():
    generate_frames(args.input_dir, args.output_dir, args.frames, args.size)
    # # Define the input and output paths
    # input_dir = '/Users/dim__gag/Desktop/videos_folder/train_root'
    # output_dir = '/Users/dim__gag/Desktop/frames'
    # 

if __name__ == '__main__':
    main()

#Example usage: python new_code/frames.py -i /Users/dim__gag/Desktop/videos_folder/train_root -o /Users/dim__gag/Desktop/frames -f 20 -s 224