# Remove .DS_Store files from the FakeTrue_DB directory and sub-directories
#
import os
import cv2
import json

def get_files_paths(dir):
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




data_dir = "/Users/dim__gag/Desktop/SASE-FE/FakeTrue_DB"
r, r_subdir, r_file = get_files_paths(data_dir)


from tqdm.auto import tqdm


# For every folder in the main directory (FakeTrue_DB) remove the jpg files
for i in tqdm(range(len(r_subdir))):
    if r_file[i].endswith(".jpg"):
        os.remove(r[i])

