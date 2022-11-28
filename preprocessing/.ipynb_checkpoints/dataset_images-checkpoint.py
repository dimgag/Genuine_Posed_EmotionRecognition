import os
import time
import numpy as np
from collections import defaultdict
import argparse


# from PIL import Image
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms



    


def move_videos(classes, list_of_videos, dataset):
    '''Function to obtain frames from videos and move them to the corresponding class folder'''
    new_base_path = os.path.join('..', '_Dataset', dataset, 'patch')

    for old_video in list_of_videos:
        participant_number = os.path.dirname(old_video).split('/')[6]

        # Validation
        if int(filter(str.isdigit, os.path.dirname(old_video).split('/')[6])) > 60:
            new_path = os.path.join(new_base_path, 'validation', classes)
            new_file_name = participant_number +'_'+ os.path.basename(old_video)
        else:
            new_path = os.path.join(new_base_path, 'training', classes)
            new_file_name = participant_number +'_'+ os.path.basename(old_video)
            print(old_video)




def list_files(dir):
	from collections import defaultdict

	videos_by_class = defaultdict(list)
	subdirs = [x[0] for x in os.walk(dir)]

	for subdir in sorted(subdirs):
		files = os.walk(subdir).next()[2]
		for file in sorted(files):
			if file.endswith('.jpeg'):
				file_name = os.path.join(subdir, file)
				videos_by_class[os.path.basename(subdir)].append( file_name )

	return videos_by_class



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# if __name__ == '__main__':


#     parser = argparse.ArgumentParser(description='Dataset preparation')


print('Starting:', time.ctime(), '\n')

# parser = argparse.ArgumentParser(description='Dataset preparation')


parser = argparse.ArgumentParser(description='Change folder order for ')
parser.add_argument('--dataset', type=str, default='patches', help='Dataset to be used')



# Split the dataset into training and validation sets
parser.add_argument('--split', type=float, default=0.8, help='Percentage of the dataset to be used for training')
parser.add_argument('--dataset', type=str, default='patches', help='Dataset to be used')
parser.add_argument('--classes', type=str, default='all', help='Classes to be used')
parser.add_argument('--frames', type=int, default=16, help='Number of frames to be used')
parser.add_argument('--size', type=int, default=224, help='Size of the images')
parser.add_argument('--channels', type=int, default=3, help='Number of channels of the images')
parser.add_argument('--seed', type=int, default=123, help='Seed for the random number generator')
parser.add_argument('--verbose', type=bool, default=True, help='Print information about the dataset')
parser.add_argument('--save', type=bool, default=True, help='Save the dataset')
parser.add_argument('--save_path', type=str, default='..', help='Path to save the dataset')
parser.add_argument('--save_name', type=str, default='dataset', help='Name to save the dataset')
parser.add_argument('--save_format', type=str, default='npy', help='Format to save the dataset')
parser.add_argument('--save_labels', type=bool, default=True, help='Save the labels')
parser.add_argument('--save_labels_path', type=str, default='..', help='Path to save the labels')
parser.add_argument('--save_labels_name', type=str, default='labels', help='Name to save the labels')
parser.add_argument('--save_labels_format', type=str, default='npy', help='Format to save the labels')
parser.add_argument('--save_classes', type=bool, default=True, help='Save the classes')
parser.add_argument('--save_classes_path', type=str, default='..', help='Path to save the classes')
parser.add_argument('--save_classes_name', type=str, default='classes', help='Name to save the classes')
parser.add_argument('--save_classes_format', type=str, default='npy', help='Format to save the classes')
parser.add_argument('--save_mean', type=bool, default=True, help='Save the mean')
parser.add_argument('--save_mean_path', type=str, default='..', help='Path to save the mean')
parser.add_argument('--save_mean_name', type=str, default='mean', help='Name to save the mean')
parser.add_argument('--save_mean_format', type=str, default='npy', help='Format to save the mean')
parser.add_argument('--save_std', type=bool, default=True, help='Save the standard deviation')
parser.add_argument('--save_std_path', type=str, default='..', help='Path to save the standard deviation')
parser.add_argument('--save_std_name', type=str, default='std', help='Name to save the standard deviation')
parser.add_argument('--save_std_format', type=str, default='npy', help='Format to save the standard deviation')
parser.add_argument('--save_min', type=bool, default=True, help='Save the minimum')
parser.add_argument('--save_min_path', type=str, default='..', help='Path to save the minimum')
