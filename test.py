import os
import json
import cv2

data_dir = "/Users/dim__gag/Desktop/SASE-FE/FakeTrue_DB"

dirlist = os.listdir(data_dir)
print(dirlist)
len(dirlist)

for folder in dirlist:
    subject_path = os.path.join(data_dir, folder)
    print(subject_path)

    # for vid in os.listdir(subject_path):
    #     filename = vid[0:2]
    #     output_path = os.path.join(subject_path, labels[vid])
    #     print(output_path)

    #     if not os.path.exists(output_path):
    #         os.makedirs(output_path)
    #     video2frames(output_path, os.path.join(subject_path, vid), filename)
    # break



# Separate the videos into folders 
# based on the emotion label


subject_path_list = []

for folder in dirlist:
    subject_path = os.path.join(data_dir, folder)
    subject_path_list.append(subject_path)
    # print(subject_path)

len(subject_path_list)


# Count the number of videos in each folder
count = 0
for i in range(len(subject_path_list)):
    # print((os.listdir(subject_path_list[i])))
    print(len(os.listdir(subject_path_list[i])))
    # print(count)



def list_files(dir):                                                                                                  
	r = []
	r_subdir = []
	r_file = []
	subdirs = [x[0] for x in os.walk(dir)]                                                                            
	for subdir in subdirs:                                                                                            
		files = os.walk(subdir).__next__()[2]                                                                             
		if (len(files) > 0):                                    
			for file in files:                                                                                        
				r.append( os.path.join(subdir, file) )   
				r_subdir.append(subdir)
				r_file.append(file)                
	return r, r_subdir, r_file

import os
import json
import cv2




# Remove .DS_Store files from the directory
dir = "/Users/dim__gag/Desktop/SASE-FE/FakeTrue_DB"

# list_files(data_dir)



# My function to remove .DS_Store files and separate the videos into folders
def list_files(dir):
    if '.DS_Store' in os.listdir(dir):
        os.remove(os.path.join(dir, '.DS_Store'))
        print("Removed .DS_Store file from main directory")
    r = []
    r_subdir = []
    r_file = []
    subdirs = [x[0] for x in os.walk(dir)]
    for subdir in subdirs:
        # if subdir contains ".DS_Store"
        if ".DS_Store" in subdir:
            print(subdir)
            # remove the file
            os.remove(subdir)
            print("Removed .DS_Store file from sub-directory")
        else:
            files = os.walk(subdir).__next__()[2]
            if (len(files) > 0):
                for file in files:
                    if file == ".DS_Store":
                        os.remove(os.path.join(subdir, file))
                        print("Removed .DS_Store file from sub-directory")
                    r.append( os.path.join
                                (subdir, file) )
                    r_subdir.append(subdir)
                    r_file.append(file)
    return r, r_subdir, r_file


list_files(data_dir)

r, r_subdir, r_file = list_files(data_dir)