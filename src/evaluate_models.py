# Load model
# methods.py
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from models.model1 import Net, EfficientNetV2M
from models.vggface import VGGFace, VGGFace2
from models.facenet import FaceNet, FaceNet_withClassifier

from data import get_datasets, get_data_loaders




# Evaluate the model
def evaluate(model, testloader, criterion, class_names):
    # Set the model to evaluation mode
    model.eval()
    # Initialize the loss and accuracy
    test_running_loss = 0.0
    test_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter +=1 
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward Pass
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            # Calculate accuracy
            _, preds = torch.max(outputs.data, 1)
            test_running_correct += (preds == labels).sum().item()
    epoch_loss = test_running_loss / counter
    epoch_acc = 100. * (test_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc






if __name__ == "__main__":
    torch.cuda.empty_cache()
	# Change the path to the model
    path = 'model.pth'
    # path = 'model.pth'
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# 1. Define the model class
    # loaded_model = Net()
    # loaded_model = VGGFace()
    # loaded_model = VGGFace2()
    # loaded_model = FaceNet()
    # loaded_model = FaceNet_withClassifier()
    loaded_model = EfficientNetV2M()
    

	# 2. Load the model parameters
	# epoch = 10
    optimizer = torch.optim.SGD(loaded_model.parameters(), lr=0.01)
	# loss = nn.CrossEntropyLoss()


    loaded_checkpoint = torch.load(path)

    loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    epoch = loaded_checkpoint['epoch']
    loss = loaded_checkpoint['loss']

    loaded_model.to(device)

    # for param in loaded_model.parameters():
    # 	print(param)
    # print(loaded_model.state_dict())

    # loaded_model.eval()

    # Get the data Local Paths
    # train_dir = "data/train"
    # test_dir = "data/test"


    dataset_train, dataset_test, dataset_classes = get_datasets()
    train_loader, test_loader = get_data_loaders(dataset_train, dataset_test)
    
    test_epoch_loss, test_epoch_acc = evaluate(loaded_model, test_loader, loss, dataset_classes)
    
    print(f"Test loss: {test_epoch_loss:.3f}, Test accuracy:{test_epoch_acc:.3f}")


    
