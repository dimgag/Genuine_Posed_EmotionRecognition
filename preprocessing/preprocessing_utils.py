import os
import cv2


def get_files_paths(dir):
    '''Function to remove .DS_Store files and get the file paths'''
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
    '''Function to extract frames from videos'''
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
                cv2.imwrite(os.path.join('data' + '/' + folder + '/' + 'real_surprise', file + dirname.split('/')[-1] + dirname.split('/')[-1] +"{:02d}.jpg".format(count) ), image)
            if file == 'N2A.MP4':
                print(file, 'real', 'angry')
                cv2.imwrite(os.path.join('data' + '/' + folder + '/' + 'real_angry', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'N2C.MP4':
                print(file, 'real', 'contempt')
                cv2.imwrite(os.path.join('data' + '/' + folder + '/' + 'real_contempt', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'N2D.MP4':
                print(file, 'real', 'disgust')
                cv2.imwrite(os.path.join('data' + '/' + folder + '/' + 'real_disgust', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'N2S.MP4':
                print(file, 'real', 'sad')
                cv2.imwrite(os.path.join('data' + '/' + folder + '/' + 'real_sad', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'N2H.MP4':
                print(file, 'real', 'happy')
                cv2.imwrite(os.path.join('data' + '/' + folder + '/' + 'real_happy', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'D2N2SUR.MP4':
                print(file, 'fake', 'surprise')
                cv2.imwrite(os.path.join('data' + '/' + folder + '/' + 'fake_surprise', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'H2N2A.MP4':
                print(file, 'fake', 'angry')
                cv2.imwrite(os.path.join('data' + '/' + folder + '/' + 'fake_angry', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'H2N2C.MP4':
                print(file, 'fake', 'contempt')
                cv2.imwrite(os.path.join('data' + '/' + folder + '/' + 'fake_contempt', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'H2N2D.MP4':
                print(file, 'fake', 'disgust')
                cv2.imwrite(os.path.join('data' + '/' + folder + '/' + 'fake_disgust', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'H2N2S.MP4':
                print(file, 'fake', 'sad')
                cv2.imwrite(os.path.join('data' + '/' + folder + '/' + 'fake_sad', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
            if file == 'S2N2H.MP4':
                print(file, 'fake', 'happy')
                cv2.imwrite(os.path.join('data' + '/' + folder + '/' + 'fake_happy', file + dirname.split('/')[-1] + "{:02d}.jpg".format(count) ), image)
        if cv2.waitKey(10) == 27:
            break

        count += 1
        frame_counter += 1
    print("Frames extracted successfully!")





def crop_faces(folder):
    emotions = os.listdir(folder)

    # Find .DS_Store and remove it from every emotion folder
    for emotion in emotions:
        if '.DS_Store' in os.listdir(folder + "/" + emotion):
            os.remove(folder + "/" + emotion + "/" + '.DS_Store')
            print("Removed .DS_Store from: ", emotion)

    if not os.path.exists(folder + "cropped_faces"):
        os.mkdir(folder + "cropped_faces")

    cropped_faces = folder + "cropped_faces"

    # Create a new directory for each emotion if not already created
    for emotion in emotions:
        if emotion not in os.listdir(cropped_faces):
            os.mkdir(cropped_faces + "/" + emotion)


    # Get the Haar Cascade classifier
    # Note: The haarcascade_frontalface_alt2.xml seems to work better that the haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')


    # Crop the faces and save them in the new directory
    for emotion in emotions:
        for image in os.listdir(folder + "/" + emotion):
            img = cv2.imread(folder + "/" + emotion + "/" + image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (25, 25), 0)
            face = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in face:
                face = img[y:y+h, x:x+w]
                cv2.imwrite(cropped_faces + "/" + emotion + "/" + image, face)

    print("Faces cropped and saved in: ", cropped_faces)

