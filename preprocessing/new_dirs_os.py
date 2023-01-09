import os 
import shutil

data_dir = "data/SASE-FE/FakeTrue_DB"
persons = os.listdir(data_dir)

emotions = ['real_surprise','real_angry','real_happy','real_sad','real_disgust','real_contempt',
                'fake_surprise','fake_angry','fake_happy','fake_sad','fake_disgust','fake_contempt']



# List files dirs


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


# 