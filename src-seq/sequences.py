import os
import glob
import cv2
import numpy as np

def get_frames_sequence(folder_path, seq_length, img_size):
    # Get all the frames in the folder
    frames = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))
    num_frames = len(frames)

    # Randomly sample a sequence of frames
    if num_frames >= seq_length:
        start_idx = np.random.randint(num_frames - seq_length + 1)
        end_idx = start_idx + seq_length
    else:
        start_idx = 0
        end_idx = num_frames

    # Load and resize the frames
    frame_sequence = []
    for i in range(start_idx, end_idx):
        frame = cv2.imread(frames[i])
        frame = cv2.resize(frame, img_size)
        frame = frame.astype(np.float32) / 255.0
        frame_sequence.append(frame)

    # Pad the sequence with zeros if necessary
    while len(frame_sequence) < seq_length:
        frame_sequence.append(np.zeros(img_size, dtype=np.float32))

    # Convert the sequence to a numpy array
    frame_sequence = np.array(frame_sequence)

    return frame_sequence


def save_sequences(input_dir, output_dir, seq_length, img_size):
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if os.path.isdir(folder_path):
            for video_file in os.listdir(folder_path):
                video_path = os.path.join(folder_path, video_file)
                output_folder = os.path.join(output_dir, folder, video_file.split('.')[0])
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                frame_sequence = get_frames_sequence(video_path, seq_length, img_size)
                np.save(os.path.join(output_folder, 'sequence.npy'), frame_sequence)



if __name__ == '__main__':
    input_dir  = 'data_sequences/val_root'
    output_dir = 'data_sequences/sequences'
    seq_length = 20
    img_size = (224, 224)

    save_sequences(input_dir, output_dir, seq_length, img_size)
