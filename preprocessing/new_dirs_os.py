import os 
import shutil
import cv2 

data_dir = "data/SASE-FE/FakeTrue_DB"
persons = os.listdir(data_dir)

emotions = ['real_surprise','real_angry','real_happy','real_sad','real_disgust','real_contempt',
                'fake_surprise','fake_angry','fake_happy','fake_sad','fake_disgust','fake_contempt']



from preprocessing_utils import get_files_paths

r, r_subdir, r_file = get_files_paths(data_dir)

print("\nWHAT is r", r[1])
print("\nWHAT is r_subdir", r_subdir[1])
print("\nWHAT is r_file", r_file[1])

# List files dirs
# print(os.listdir(data_dir))


'''
if ".DS_Store" in persons:
    persons.remove(".DS_Store")

print(persons)

print(len(persons), "persons in the dataset")


# Make dir
if not os.path.exists("data/train1"):
    os.mkdir("data/train1")

if not os.path.exists("data/test1"):
    os.mkdir("data/test1")

# # Make emotion dirs
# for emotion in os.listdir(data_dir):
#     os.mkdir("data/train1/" + emotion)
#     os.mkdir("data/test1/" + emotion)


# print(os.listdir(data_dir))

for emotion in emotions:
    if not os.path.exists("data/train1/" + emotion):
        os.mkdir("data/train1/" + emotion)



# Make Frames directory 
if not os.path.exists("data/frames"):
    os.mkdir("data/frames")

# make persons directories in frames
for person in persons:
    if not os.path.exists("data/frames/" + person):
        os.mkdir("data/frames/" + person)




# Generate the frames from videos and move them to frames directories 
for person in persons:
    for emotion in emotions:
        videos = os.listdir(data_dir + "/" + person + "/" + emotion)
        for video in videos:
            if ".DS_Store" in video:
                videos.remove(".DS_Store")
        for video in videos:
            os.system("ffmpeg -i " + data_dir + "/" + person + "/" + emotion + "/" + video + " -vf fps=1 data/frames/" + person + "/" + video[:-4] + "_%d.jpg")
    print("Frames Generated")





# Crop the frames







# for first 40 persons 
for person in persons[:40]:
    for emotion in emotions:
        images = os.listdir(data_dir + "/" + person + "/" + emotion)
        for image in images:
            if ".DS_Store" in image:
                images.remove(".DS_Store")
        for image in images[int(len(images))]:
            shutil.move(data_dir + "/" + person + "/" + emotion + "/" + image, "data/train1/" + emotion + "/" + image)
    print("Train Directory Created")

# for last 40 persons
for person in persons[40:]:
    for emotion in emotions:
        images = os.listdir(data_dir + "/" + person + "/" + emotion)
        for image in images:
            if ".DS_Store" in image:
                images.remove(".DS_Store")
        for image in images[int(len(images))]:
            shutil.move(data_dir + "/" + person + "/" + emotion + "/" + image, "data/test1/" + emotion + "/" + image)
    print("Test Directory Created")


# '''