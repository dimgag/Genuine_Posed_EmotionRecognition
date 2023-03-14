import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class VideoDataset(Dataset):
    def __init__(self, sequences_folder, seq_length):
        self.sequences_folder = sequences_folder
        self.seq_length = seq_length
        self.sequences = []
        self.labels = []

        # walk through the sequences folder and load all the sequences
        for dirpath, dirnames, filenames in os.walk(sequences_folder):
            # print(dirpath)
            # print(dirnames)
            # print(filenames)
            for filename in filenames:
                if filename.endswith('.npy'):
                    sequence_path = os.path.join(dirpath, filename)
                    sequence = np.load(sequence_path, allow_pickle=True)
                    self.sequences.append(sequence)
                    
                    # get label from the folder name
                    label = os.path.basename(os.path.dirname(dirpath))
                    self.labels.append(label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if len(sequence) > self.seq_length:
            start_idx = np.random.randint(len(sequence) - self.seq_length + 1)
            end_idx = start_idx + self.seq_length
            sequence = sequence[start_idx:end_idx]
        elif len(sequence) < self.seq_length:
            sequence = np.pad(sequence, ((0, self.seq_length - len(sequence)), (0, 0), (0, 0), (0, 0)), 'constant')
            
        return torch.tensor(sequence), label



def remove_ds_store(directory):
    for dirpath, dirnames, filenames in os.walk(directory):
        # Remove .DS_Store files in the current directory
        if ".DS_Store" in filenames:
            os.remove(os.path.join(dirpath, ".DS_Store"))
            print(f"Removed .DS_Store file from {dirpath}")

        # Recursively remove .DS_Store files in subdirectories
        for dirname in dirnames:
            subdir = os.path.join(dirpath, dirname)
            remove_ds_store(subdir)

            
            
def get_data_loaders(train_sequences_folder, val_sequences_folder, seq_length, batch_size, num_workers):
    train_dataset = VideoDataset(train_sequences_folder, seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    val_dataset = VideoDataset(val_sequences_folder, seq_length)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_dataloader, val_dataloader
    
    

    
    

if __name__ == '__main__':
    # Set the parameters
    # sequences_folder = 'data_sequences/train_seq'
    # remove_ds_store(sequences_folder)

#     seq_length = 20
#     batch_size = 1
#     num_workers = 2

    # Create the dataset and dataloader
    # dataset = VideoDataset(sequences_folder, seq_length)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # print dataset labels
    # unique_labels = list(set(dataset.labels))
    # print(unique_labels)
    
    # print dataset length
    # print(f"Dataset length: {len(dataset)}")
    
    # print dataloader length
    # print(f"Dataloader length: {len(dataloader)}")
    
    
    train_dataloader, val_dataloader = get_data_loaders('data_sequences/train_seq',
                                                        'data_sequences/val_seq',
                                                        seq_length=20,
                                                        batch_size=1,
                                                        num_workers=2)
    
    print(f"Train Dataloader length: {len(train_dataloader)}")
    print(f"Validatio Dataloader length: {len(val_dataloader)}")
    




