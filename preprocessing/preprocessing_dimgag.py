import os 
import json
import cv2

# Create a new directory for the frames
def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Created directory: ", dir)
    else:
        print("Directory already exists: ", dir)



# Function to remove .DS_Store files and separate the videos into folders
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

# Get the paths of the videos
# paths, subdirs, files = get_files_paths('/Users/dim__gag/Desktop/SASE-FE/FakeTrue_DB')


# Function to extract frames from videos
def video2frames(dir, dirname, file):
    vidcap = cv2.VideoCapture(dir)
    '''
	Get length of the videos in frames:
		- Get frames from 50% of the video's length until 90% of the length
	'''
    length_of_video = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, image = vidcap.read()
    

    new_dir = dir.replace('patches', 'frames3D').split('.MP4')[0]
    print(new_dir)

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
                cv2.imwrite(os.path.join('/Users/dim__gag/Desktop/SASE-FE/frames/real/surprise', file + 'frame%d.jpg' % count), image)
            if file == 'N2A.MP4':
                print(file, 'real', 'angry')
                cv2.imwrite(os.path.join('/Users/dim__gag/Desktop/SASE-FE/frames/real/angry', file + 'frame%d.jpg' % count), image)
            if file == 'N2C.MP4':
                print(file, 'real', 'contempt')
                cv2.imwrite(os.path.join('/Users/dim__gag/Desktop/SASE-FE/frames/real/contempt', file + 'frame%d.jpg' % count), image)
            if file == 'N2D.MP4':
                print(file, 'real', 'disgust')
                cv2.imwrite(os.path.join('/Users/dim__gag/Desktop/SASE-FE/frames/real/disgust', file + 'frame%d.jpg' % count), image)
            if file == 'N2S.MP4':
                print(file, 'real', 'sad')
                cv2.imwrite(os.path.join('/Users/dim__gag/Desktop/SASE-FE/frames/real/sad', file + 'frame%d.jpg' % count), image)
            if file == 'N2H.MP4':
                print(file, 'real', 'happy')
                cv2.imwrite(os.path.join('/Users/dim__gag/Desktop/SASE-FE/frames/real/happy', file + 'frame%d.jpg' % count), image)
            if file == 'D2N2SUR.MP4':
                print(file, 'fake', 'surprise')
                cv2.imwrite(os.path.join('/Users/dim__gag/Desktop/SASE-FE/frames/fake/surprise', file + 'frame%d.jpg' % count), image)
            if file == 'H2N2A.MP4':
                print(file, 'fake', 'angry')
                cv2.imwrite(os.path.join('/Users/dim__gag/Desktop/SASE-FE/frames/fake/angry', file + 'frame%d.jpg' % count), image)
            if file == 'H2N2C.MP4':
                print(file, 'fake', 'contempt')
                cv2.imwrite(os.path.join('/Users/dim__gag/Desktop/SASE-FE/frames/fake/contempt', file + 'frame%d.jpg' % count), image)
            if file == 'H2N2D.MP4':
                print(file, 'fake', 'disgust')
                cv2.imwrite(os.path.join('/Users/dim__gag/Desktop/SASE-FE/frames/fake/disgust', file + 'frame%d.jpg' % count), image)
            if file == 'H2N2S.MP4':
                print(file, 'fake', 'sad')
                cv2.imwrite(os.path.join('/Users/dim__gag/Desktop/SASE-FE/frames/fake/sad', file + 'frame%d.jpg' % count), image)
            if file == 'S2N2H.MP4':
                print(file, 'fake', 'happy')
                cv2.imwrite(os.path.join('/Users/dim__gag/Desktop/SASE-FE/frames/fake/happy', file + 'frame%d.jpg' % count), image)
        if cv2.waitKey(10) == 27:
            break

        count += 1
        frame_counter += 1
        


if __name__ == '__main__':
    # Data directory:
    data_dir = "/Users/dim__gag/Desktop/SASE-FE/FakeTrue_DB"

    # Create a new directory for the frames
    create_dir('/Users/dim__gag/Desktop/SASE-FE/frames')
    frames_dir = '/Users/dim__gag/Desktop/SASE-FE/frames'

    # Create subdirectories for the frames real and fake
    create_dir('/Users/dim__gag/Desktop/SASE-FE/frames/real')
    create_dir('/Users/dim__gag/Desktop/SASE-FE/frames/fake')

    # Emotions
    emotions = ['surprise', 'angry', 'happy', 'sad', 'disgust', 'contempt']

    # Create subdirectories for the emotions
    for emotion in emotions:
        create_dir('/Users/dim__gag/Desktop/SASE-FE/frames/real/' + emotion)
        create_dir('/Users/dim__gag/Desktop/SASE-FE/frames/fake/' + emotion)

    # Get the paths of the videos
    r, r_subdir, r_file = get_files_paths(data_dir)
    
    # Capitalize all video names
    for i in range(len(r_file)):
        r_file[i] = r_file[i].upper()
    
    # Extract frames from videos
    for video, dirname, file in zip(r, r_subdir, r_file):
        video2frames(video, dirname, file)
    








