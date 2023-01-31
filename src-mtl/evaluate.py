import os
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import HydraNet, ChimeraNet
from dataset import SASEFE_MTL, SASEFE_MTL_TEST

def evaluate(model, testloader):
    model.eval()
    print("Validating model...")
    
    # Define the loss functions.
    emotion_loss = nn.CrossEntropyLoss() # Includes Softmax
    real_fake_loss = nn.BCELoss() # Doesn't include Softmax
    # real_fake_loss = nn.BCEWithLogitsLoss() 

    Sig = nn.Sigmoid()
    total_validation_loss = 0.0
    emotion_validation_acc = 0
    real_fake_validation_acc = 0 
    counter = 0
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter +=1
            inputs = data["image"].to(device)

            real_fake_label = data["real_fake"].to(device) 
            emotion_label = data["emotion"].to(device)

            # Forward pass
            real_fake_output, emotion_output = model(inputs)

            # Calculate the Loss
            loss_1 = emotion_loss(emotion_output, emotion_label)
            loss_2 = real_fake_loss(Sig(real_fake_output), real_fake_label.unsqueeze(1).float())
            loss = loss_1 + loss_2
            total_validation_loss += loss

            # Calculate Accuracy
            _, emo_preds = torch.max(emotion_output.data, 1)
            emotion_validation_acc += (emo_preds == emotion_label).sum().item()
            
            _, rf_preds = torch.max(real_fake_output.data, 1)
            real_fake_validation_acc += (rf_preds == real_fake_label).sum().item()
            

        epoch_loss = total_validation_loss / counter
        epoch_acc_emotion = 100. * (emotion_validation_acc / len(testloader.dataset))
        epoch_acc_real_fake = 100. * (real_fake_validation_acc / len(testloader.dataset))
    
    return epoch_loss, epoch_acc_emotion, epoch_acc_real_fake


if __name__ == "__main__":
    print("Evaluate model")
    torch.cuda.empty_cache()
    # Load the model.pth
    path = 'experiments/last-experiment-sigmoid_as_last_layer/model.pth'

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the model class
    # loaded_model = HydraNet().to(device)
    loaded_model = ChimeraNet().to(device)

    # Define the optimizer
    optimizer = torch.optim.SGD(loaded_model.parameters(), lr=1e-4, momentum=0.09)

    # Define the loss functions.
    emotion_loss = nn.CrossEntropyLoss()
    real_fake_loss = nn.BCELoss()

    # Load the checkpoint
    loaded_checkpoint = torch.load(path)

    # 
    loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    epoch = loaded_checkpoint['epoch']

    # Load the data

    test_dir = "data_mtl/test"

    test_image_paths = os.listdir(test_dir)


    # Get the dataset class
    test_dataset = SASEFE_MTL_TEST(test_image_paths)

    # Get the dataloaders
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    # Evaluate the model
    epoch_loss, epoch_acc_emotion, epoch_acc_real_fake = evaluate(loaded_model, test_dataloader)
    print(f"Test loss: {epoch_loss:.3f}, Test Emotion acc: {epoch_acc_emotion:.3f}, Test Real/Fake acc: {epoch_acc_real_fake:.3f}")

    


