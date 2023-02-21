# Create MultiTaskDataset class for the temporal data to work with TimeSformer model.
import os
import av
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader

class MultiTaskDataset(data.Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

        # Define the list of emotions
        self.emotions = ['Angry', 'Contempt', 'Disgust', 'Happy', 'Sad', 'Surprise']

        # Create a dictionary to map the emotion labels to integers
        self.emotion_to_int = {emotion: i for i, emotion in enumerate(self.emotions)}

        # Define the list of video filenames
        self.video_filenames = []
        for participant in range(1, 51):
            for real_fake in ['Real', 'Fake']:
                for emotion in self.emotions:
                    video_filename = f'Participant{participant}/{real_fake}/{emotion}.MP4'
                    self.video_filenames.append(video_filename)

    def __len__(self):
        return len(self.video_filenames)

    def __getitem__(self, index):
        # Load the video file
        video_filename = self.video_filenames[index]
        video_path = os.path.join(self.root_dir, video_filename)
        video_data = self._load_video(video_path)

        # Get the emotion and real/fake labels from the video filename
        real_fake, emotion = video_filename.split('/')[2:]
        real_fake_target = 1 if real_fake == 'Real' else 0
        emotion_target = self.emotion_to_int[emotion]

        target = (real_fake_target, emotion_target)

        if self.transform is not None:
            video_data = self.transform(video_data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return video_data, target

    def _load_video(self, video_path):
        # Load the video file using PyAV
        container = av.open(video_path)
        video_data = []

        # Iterate over the frames of the video and convert to RGB format
        for frame in container.decode(video=0):
            image = frame.to_image()
            image_data = np.array(image)[:, :, ::-1]  # Convert BGR to RGB
            video_data.append(image_data)

        # Convert the list of frames to a PyTorch tensor
        video_data = torch.tensor(video_data)

        return video_data



from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.RandomHorizontalFlip(p=0.5), #Data Augmentation
                # transforms.RandomRotation(35), #Data Augmentation
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4270, 0.3508, 0.2971], std=[0.1844, 0.1809, 0.1545])
                ])

    # Define the dataset
    dataset = MultiTaskDataset(data_path='data_temporal', transform=transform)

    # Split the dataset into train and test sets
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

    # Batch size
    batch_size = 32

    # create the data loaders for train and test sets
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Print the number of videos in the train and test sets
    print(f'Train set size: {len(train_set)}')
    print(f'Test set size: {len(test_set)}')

