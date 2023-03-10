# Python script to resize a video to the given short edge size -256.
# python video_resize.py -i /Users/dim__gag/Desktop/data_temporal/train_root/fake_angry -o /Users/dim__gag/Desktop/data_temporal_resized
import os
import cv2
import argparse

parser = argparse.ArgumentParser(description="Video Resize")
parser.add_argument('-i', '--input_dir', type=str, default = None)
parser.add_argument('-o', '--output_dir', type=str, default = None)
parser.add_argument('--size', '--short_edge_size', type=int, default = 256 , help='face size nxn')
parser.add_argument('--fps', '--frames', type=int, default = 100 , help='fps of the output video')

args = parser.parse_args()


def video_resizer(input_dir, output_dir):

    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mp4"):
                input_path = os.path.join(subdir, file)
                relpath = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relpath)

                # make the output directory if it does not exist
                outdir = os.path.dirname(output_path)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                cap = cv2.VideoCapture(input_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                aspect_ratio = width / height
                if width > height:
                    new_width = args.size
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = args.size
                    new_width = int(new_height * aspect_ratio)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, args.fps, (new_width, new_height))

                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret==True:
                        frame_resized = cv2.resize(frame, (new_width, new_height))
                        out.write(frame_resized)
                    else:
                        break
                cap.release()
                out.release()


if __name__ == "__main__":
    video_resizer(args.input_dir, args.output_dir)
