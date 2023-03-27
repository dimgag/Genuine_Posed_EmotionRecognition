import os
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import HydraNet, ChimeraNet, ChimeraNetV2 
from dataset import SASEFE_MTL, SASEFE_MTL_TEST
from utils import ConfusionMatrix_MT
# from confusion_matrix import CM


def evaluate(model, testloader):
    model.eval()
    print("Evaluating the model...")
    
    # Define the loss functions.
    emotion_loss = nn.CrossEntropyLoss()
    real_fake_loss = nn.CrossEntropyLoss()

    total_validation_loss = 0.0
    emotion_validation_acc = 0
    real_fake_validation_acc = 0 
    overall_validation_acc = 0
    both_correct = 0
    counter = 0
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter +=1
            inputs = data["image"].to(device)

            real_fake_label = data["real_fake"].to(device) 
            emotion_label = data["emotion"].to(device)

            # Forward pass
            real_fake_output, emotion_output = model(inputs)
            # ------------------------------------------------------------
            # Calculate the Loss
            loss_1 = emotion_loss(emotion_output, emotion_label)
            loss_2 = real_fake_loss(real_fake_output, real_fake_label)
            # IDEA 1: TOTAL LOSS = SUM
            # loss = loss_1 + loss_2
            # total_training_loss += loss
            # loss = loss_1 + loss_2
            # total_validation_loss += loss
            # ------------------------------------------------------------
            # IDEA 2: TOTAL LOSS = PRECISION_1 * LOSS_1 + PRECISION_2 * LOSS_2
            # Calculate precision for emotions
            _, emo_preds = torch.max(emotion_output.data, 1)
            emo_true_positives = (emo_preds == emotion_label).sum().item()
            emo_total_predicted_positives = (emo_preds == emo_preds).sum().item()
            precision_emotions = emo_true_positives / emo_total_predicted_positives
            # Calculate precision for emotions and real/fake 
            _, rf_preds = torch.max(real_fake_output.data, 1)
            rf_true_positives = (rf_preds == real_fake_label).sum().item()
            rf_total_predicted_positives = (rf_preds == rf_preds).sum().item()
            precision_real_fake = rf_true_positives / rf_total_predicted_positives
            # Calculation of combined loss
            loss_emotions = precision_emotions * loss_1
            loss_real_fake = precision_real_fake * loss_2
            loss = loss_emotions + loss_real_fake
            total_validation_loss += loss
            # ------------------------------------------------------------
            
            # Calculate Accuracy for Emotions
            emotion_validation_acc += (emo_preds == emotion_label).sum().item()
            
            # Calculate Accuracy for Real / fake
            real_fake_validation_acc += (rf_preds == real_fake_label).sum().item()
            
            # Calculate Overall Accuracy Count number of samples for which both tasks are correctly classified
            both_correct += ((emo_preds == emotion_label) & (rf_preds == real_fake_label)).sum().item()

        

        epoch_loss = total_validation_loss / counter
        epoch_acc_emotion = 100. * (emotion_validation_acc / len(testloader.dataset))
        epoch_acc_real_fake = 100. * (real_fake_validation_acc / len(testloader.dataset))
        overall_validation_acc = 100. * (both_correct / len(testloader.dataset))
    
    return epoch_loss, epoch_acc_emotion, epoch_acc_real_fake, overall_validation_acc


if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Load the model.pth - change the path to the model you want to evaluate
    path = 'experiments/experiments_MTL/exp1-MTL/model.pth'

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    
    # Define the model class
    loaded_model = HydraNet().to(device)
    # loaded_model = ChimeraNet().to(device)
    # loaded_model = ChimeraNetV2().to(device)

    # Define the optimizer
    optimizer = torch.optim.SGD(loaded_model.parameters(), lr=1e-4, momentum=0.09)

    # Define the loss functions.
    emotion_loss = nn.CrossEntropyLoss()
    # real_fake_loss = nn.BCELoss()
    real_fake_loss = nn.CrossEntropyLoss()

    # Load the checkpoint
    loaded_checkpoint = torch.load(path, map_location=torch.device('cpu'))

    # Load the model state
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
    epoch_loss, epoch_acc_emotion, epoch_acc_real_fake, overall_validation_acc = evaluate(loaded_model, test_dataloader)
    print(f"Test loss: {epoch_loss:.3f}, Test Emotion acc: {epoch_acc_emotion:.3f}, Test Real/Fake acc: {epoch_acc_real_fake:.3f}, Overall Accuracy: {overall_validation_acc:.3f}")
    
    # Confussion matrix 
    # Get the classes of the dataset to generate the confusion matrix
#     real_fake_classes = test_dataset.real_fakes
#     real_fake_classes = np.unique(real_fake_classes)
#     convert_dict = {0: 'fake', 1: 'real'}
#     real_fake_classes = [convert_dict.get(i, i) for i in real_fake_classes]

#     emotion_classes = test_dataset.emotions
#     emotion_classes = np.unique(emotion_classes)
#     convert_dict = {0: 'happy', 1: 'sad', 2: 'surprise', 3: 'disgust', 4: 'contempt', 5: 'angry'}
#     emotion_classes = [convert_dict.get(i, i) for i in emotion_classes]

    # ConfusionMatrix_MT(loaded_model, test_dataloader, real_fake_classes, emotion_classes)
