# After all the videos were downloaded, 
# resize the video to the short edge size of 256, 
# then prepare the csv files for training, validation, and testing set as train.csv, val.csv, test.csv. 
# The format of the csv file is:

# Resize the videos to the short edge size of 256
# python resize_videos.py --source_dir videos --output_dir videos_resized

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import sys
from multiprocessing import Pool


def resize_videos(vid_item, args):
    """Generate resized video cache.
    Args:
        vid_item (list): Video item containing video full path,
            video relative path.
    Returns:
        bool: Whether generate video cache successfully.
    """
    full_path, vid_path = vid_item
    # Change the output video extension to .mp4 if '--to-mp4' flag is set
    if args.to_mp4:
        vid_path = vid_path.split('.')
        assert len(vid_path) == 2, \
            f"Video path '{vid_path}' contain more than one dot"
        vid_path = vid_path[0] + '.mp4'
    out_full_path = osp.join(args.out_dir, vid_path)
    dir_name = osp.dirname(vid_path)
    out_dir = osp.join(args.out_dir, dir_name)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    result = os.popen(
        f'ffprobe -hide_banner -loglevel error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 {full_path}'  # noqa:E501
    )
    w, h = [int(d) for d in result.readline().rstrip().split(',')]
    if w > h:
        cmd = (f'ffmpeg -hide_banner -loglevel error -i {full_path} '
               f'-vf {"mpdecimate," if args.remove_dup else ""}'
               f'scale=-2:{args.scale} '
               f'{"-vsync vfr" if args.remove_dup else ""} '
               f'-c:v libx264 {"-g 16" if args.dense else ""} '
               f'-an {out_full_path} -y')
    else:
        cmd = (f'ffmpeg -hide_banner -loglevel error -i {full_path} '
               f'-vf {"mpdecimate," if args.remove_dup else ""}'
               f'scale={args.scale}:-2 '
               f'{"-vsync vfr" if args.remove_dup else ""} '
               f'-c:v libx264 {"-g 16" if args.dense else ""} '
               f'-an {out_full_path} -y')
    os.popen(cmd)
    print(f'{vid_path} done')
    sys.stdout.flush()
    return True


def run_with_args(item):
    vid_item, args = item
    return resize_videos(vid_item, args)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate the resized cache of original videos')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('out_dir', type=str, help='output video directory')
    parser.add_argument(
        '--dense',
        action='store_true',
        help='whether to generate a faster cache')
    parser.add_argument(
        '--level',
        type=int,
        choices=[1, 2],
        default=2,
        help='directory level of data')
    parser.add_argument(
        '--remove-dup',
        action='store_true',
        help='whether to remove duplicated frames')
    parser.add_argument(
        '--ext',
        type=str,
        default='mp4',
        choices=['avi', 'mp4', 'webm', 'mkv'],
        help='video file extensions')
    parser.add_argument(
        '--to-mp4',
        action='store_true',
        help='whether to output videos in mp4 format')
    parser.add_argument(
        '--scale',
        type=int,
        default=256,
        help='resize image short side length keeping ratio')
    parser.add_argument(
        '--num-worker', type=int, default=8, help='number of workers')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not osp.isdir(args.out_dir):
        print(f'Creating folder: {args.out_dir}')
        os.makedirs(args.out_dir)

    print('Reading videos from folder: ', args.src_dir)
    print('Extension of videos: ', args.ext)
    fullpath_list = glob.glob(args.src_dir + '/*' * args.level + '.' +
                              args.ext)
    done_fullpath_list = glob.glob(args.out_dir + '/*' * args.level + args.ext)
    print('Total number of videos found: ', len(fullpath_list))
    print('Total number of videos transfer finished: ',
          len(done_fullpath_list))
    if args.level == 2:
        vid_list = list(
            map(
                lambda p: osp.join(
                    osp.basename(osp.dirname(p)), osp.basename(p)),
                fullpath_list))
    elif args.level == 1:
        vid_list = list(map(osp.basename, fullpath_list))
    pool = Pool(args.num_worker)
    vid_items = zip(fullpath_list, vid_list)
    pool.map(run_with_args, [(item, args) for item in vid_items])







'''
This code is a Python script that can be used to generate resized video cache from a source video directory. To use this code, you will need to do the following steps:

Install required packages: This code uses the argparse, glob, os, os.path, sys, and multiprocessing Python packages. If you do not have these packages installed, you will need to install them before running the script. 
You can use the following command to install these packages:

`pip install argparse glob2 os-sys multiprocessing`

Set up the source and output video directories: 
You will need to specify the source video directory and output video directory by providing their paths as command-line arguments when running the script. 
The source video directory is the directory that contains the original videos that you want to resize, and the output video directory is the directory where the resized videos will be stored.

Run the script: Once you have set up the source and output video directories, you can run the script by typing the following command in your terminal or command prompt:

Copy code
python resize_videos.py source_video_directory output_video_directory
You can also provide various optional command-line arguments to customize the resizing process. For example, you can use the --dense flag to generate a faster cache, the --remove-dup flag to remove duplicated frames, and the --to-mp4 flag to output videos in mp4 format.

Wait for the script to finish: The script will take some time to resize all the videos in the source video directory and store them in the output video directory. Once the script is finished, you can check the output video directory to make sure that all the resized videos are there.

Note: It is recommended to use this script on a computer with multiple cores or CPUs to speed up the resizing process. You can adjust the --num-worker flag to specify the number of worker processes to use for parallel processing.

Example: 
python src-temporal/resize_videos.py /Users/dim__gag/Desktop/test /Users/dim__gag/Desktop/test --dense --level 1 --remove-dup --ext mp4 --to-mp4 --scale 256 --num-worker 8
'''