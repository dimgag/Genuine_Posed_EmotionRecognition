import os
import glob
import json

VIDEO_FORMATS = ('.mp4', '.avi', '.mov', '.mkv')

def convert_to_kinetics_format(dataset_dir, output_file):
    """
    Convert video dataset to Kinetics format.

    Args:
    dataset_dir (str): Directory containing videos.
    output_file (str): Path to output file.
    """
    # Find all video files in the dataset directory
    video_files = []
    for format in VIDEO_FORMATS:
        video_files.extend(glob.glob(os.path.join(dataset_dir, '**', f'*{format}'), recursive=True))
    
    # Convert video files to Kinetics format
    kinetics_data = []
    for video_file in video_files:
        # label = os.path.basename(os.path.dirname(os.path.dirname(video_file)))
        # 
        label = os.path.basename(os.path.dirname(video_file))
        video_id = os.path.splitext(os.path.basename(video_file))[0]
        kinetics_data.append({
            'path': video_file,
            'label': label,
            'id': video_id
        })
    
    # Save Kinetics data to output file
    with open(output_file, 'w') as f:
        json.dump(kinetics_data, f, indent=4)

if __name__ == '__main__':
    train_dir = 'data_temporal/train_root'
    train_output_file = 'data_temporal/train_root.json'
    convert_to_kinetics_format(train_dir, train_output_file)

    val_dir = 'data_temporal/val_root'
    val_output_file = 'data_temporal/val_root.json'
    convert_to_kinetics_format(val_dir, val_output_file)