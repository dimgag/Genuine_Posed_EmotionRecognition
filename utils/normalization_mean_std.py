# Find the mean and standard deviation of the dataset color channels to use it in the normalization.

import torch 

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

#  Try it out
# dataset_train, dataset_test, dataset_classes = get_datasets()
# Load the training and Test data loaders
# train_loader, test_loader = get_data_loaders(dataset_train, dataset_test)

# mean, std = get_mean_and_std(train_loader)

# print("Mean is: ", mean)
# print("\nStd is: ", std)
