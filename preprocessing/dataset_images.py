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




    