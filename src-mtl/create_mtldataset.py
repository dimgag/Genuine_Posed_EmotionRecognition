# I want to generate frames from videos and then use them to train the model
# The names of the frames should be in the following format:
# [emotion]_[real/fake]_[frame_number].jpg

import os 
import cv2
import shutil

## Create the forders for train and test

# data_dir = "data/SASE-FE/FakeTrue_DB"
# persons = os.listdir(data_dir)
# for person in persons:
#     if ".DS_Store" in person:
#         persons.remove(".DS_Store")
#     for person in persons[:40]:
#         # copy persons folder to train
#         if not os.path.exists("data_mtl/train/" + person):
#             shutil.copytree(data_dir + "/" + person, "data_mtl/train/" + person)
#     print("Copied train files done")
    
#     for person in persons[40:50]:
#         # copy persons folder to test
#         if not os.path.exists("data/test/" + person):
#             shutil.copytree(data_dir + "/" + person, "data_mtl/test/" + person)
#     print("Copied test files done")




def get_files_paths(dir):
  #  Function to remove .DS_Store files and get the file paths
    if '.DS_Store' in os.listdir(dir):
        os.remove(os.path.join(dir, '.DS_Store'))
        print("Remove .DS_Store file from main directory")
    r = []
    r_subdir = []
    r_file = []
    subdirs = [x[0] for x in os.walk(dir)]
    for subdir in subdirs:
        if '.DS_Store' in subdir:
            os.remove(subdir)
            print("Remove .DS_store file from sub-directory", subdir)
        else:
            files = os.walk(subdir).__next__()[2]
            if (len(files) > 0):
                for file in files:
                    if file == '.DS_Store':
                        os.remove(os.path.join(subdir, file))
                        print("Removed .DS_Store file from sub-directory")
                    r.append(os.path.join(subdir, file))
                    r_subdir.append(subdir)
                    r_file.append(file)
    r_file = [file.upper() for file in r_file]
    return r, r_subdir, r_file


def video2frames(dir, dirname, file, folder):
    # Function to extract frames from videos
    vidcap = cv2.VideoCapture(dir)
    length_of_video = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, image = vidcap.read()

    new_dir = dir.replace('patches', 'frames3D').split('.MP4')[0]
    print("New directory for the frames: ", new_dir)
    print(os.path.basename(new_dir))

    count = 0
    frame_counter = 0
    vid = []
    while success:
        success, image = vidcap.read()

        if frame_counter > 16:
            frame_counter = 0
            print(file)
            vid = []

        if count > int(length_of_video*.6) and count < int(length_of_video*.9):
            vid.append(dirname)
            if file == 'N2SUR.MP4':            
                print(file, 'real', 'surprise')
                cv2.imwrite(os.path.join('data_mtl' + '/' + folder,  'real_surprise_' + file + dirname.split('/')[-1] + dirname.split('/')[-1] +"{:02d}.jpg".format(count) ), image)
            if file == 'N2A.MP4':
                print(file, 'real', 'angry')
                cv2.imwrite(os.path.join('data_mtl' + '/' + folder, 'real_angry_' + file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'N2C.MP4':
                print(file, 'real', 'contempt')
                cv2.imwrite(os.path.join('data_mtl' + '/' + folder, 'real_contempt_' + file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'N2D.MP4':
                print(file, 'real', 'disgust')
                cv2.imwrite(os.path.join('data_mtl' + '/' + folder, 'real_disgust_' + file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'N2S.MP4':
                print(file, 'real', 'sad')
                cv2.imwrite(os.path.join('data_mtl' + '/' + folder, 'real_sad_' + file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'N2H.MP4':
                print(file, 'real', 'happy')
                cv2.imwrite(os.path.join('data_mtl' + '/' + folder, 'real_happy_'+ file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'D2N2SUR.MP4':
                print(file, 'fake', 'surprise')
                cv2.imwrite(os.path.join('data_mtl' + '/' + folder, 'fake_surprise_' + file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'H2N2A.MP4':
                print(file, 'fake', 'angry')
                cv2.imwrite(os.path.join('data_mtl' + '/' + folder, 'fake_angry_' + file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'H2N2C.MP4':
                print(file, 'fake', 'contempt')
                cv2.imwrite(os.path.join('data_mtl' + '/' + folder, 'fake_contempt_' + file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'H2N2D.MP4':
                print(file, 'fake', 'disgust')
                cv2.imwrite(os.path.join('data_mtl' + '/' + folder, 'fake_disgust_' + file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'H2N2S.MP4':
                print(file, 'fake', 'sad')
                cv2.imwrite(os.path.join('data_mtl' + '/' + folder, 'fake_sad_' + file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'S2N2H.MP4':
                print(file, 'fake', 'happy')
                cv2.imwrite(os.path.join('data_mtl' + '/' + folder, 'fake_happy_' + file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
        if cv2.waitKey(10) == 27:
            break

        count += 1
        frame_counter += 1
    print("Frames extracted successfully!")


def crop_faces(folder):
# Find .DS_Store and remove it from every emotion folder
    if '.DS_Store' in os.listdir(folder):
        os.remove(folder + "/" + '.DS_Store')
        print("Removed .DS_Store")

    if not os.path.exists(folder + "/" + "cropped_faces"):
        os.mkdir(folder + "/" + "cropped_faces")


    cropped_faces = folder + "/" + "cropped_faces"


    # Get the Haar Cascade classifier
    # Note: The haarcascade_frontalface_alt2.xml seems to work better that the haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')


    # Crop the faces and save them in the new directory
    for image in os.listdir(folder):
        if image != 'cropped_faces':
            # print("Cropping Image:", image)
            img = cv2.imread(folder + "/" + image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (25, 25), 0)
            face = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in face:
                face = img[y:y+h, x:x+w]
                cv2.imwrite(cropped_faces + "/" + image, face)

    print("Faces cropped and saved in: ", cropped_faces)





def main():
    data_dir = "data/SASE-FE/FakeTrue_DB"
    persons = os.listdir(data_dir)


    # Train folder
    train_prep = "data_mtl/train"
#     r, r_subdir, r_file = get_files_paths(train_prep)

#     for video, dirname, file in zip(r, r_subdir, r_file):
#         video2frames(video, dirname, file, 'train')
    if '.DS_Store' in os.listdir('data_mtl/train'):
        os.remove('data_mtl/train' + "/" + '.DS_Store')
        print("Removed .DS_Store")
    
    for file in os.listdir(train_prep):
        os.rename(os.path.join(train_prep, file), os.path.join(train_prep, file.replace(' ', '')))
    
    crop_faces(train_prep)

    #######################################################
    # Test folder
#     test_prep = "data_mtl/test"
# #     r, r_subdir, r_file = get_files_paths(test_prep)

# #     for video, dirname, file in zip(r, r_subdir, r_file):
# #         video2frames(video, dirname, file, 'test')
#     if '.DS_Store' in os.listdir('data_mtl/test'):
#         os.remove('data_mtl/test' + "/" + '.DS_Store')
#         print("Removed .DS_Store")
    
#     for file in os.listdir(test_prep):
#         os.rename(os.path.join(test_prep, file), os.path.join(test_prep, file.replace(' ', '')))
#     crop_faces(test_prep)



if __name__ == '__main__':
    main()
