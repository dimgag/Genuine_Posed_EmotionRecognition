# Load model
# methods.py
import torch
import torch.nn as nn
from models.model1 import Net
from data import get_datasets, get_data_loaders

# Evaluate the model
def evaluate(model, test_loader, criterion):
	# Set the model to evaluation mode
	model.eval()
	# Initialize the loss and accuracy
	loss = 0
	accuracy = 0
	# Iterate over the test data and generate predictions
	for images, labels in test_loader:
		# Move input and label tensors to the default device
		images, labels = images.to(device), labels.to(device)
		# Get the log probabilities from the model
		logps = model(images)
		# Calculate the loss
		loss += criterion(logps, labels)
		# Calculate the accuracy
		ps = torch.exp(logps)
		top_p, top_class = ps.topk(1, dim=1)
		equals = top_class == labels.view(*top_class.shape)
		accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
	# Return the average loss and accuracy
	return loss/len(test_loader), accuracy/len(test_loader)







if __name__ == "__main__":
	# Change the path to the model
	path =  '/Users/dim__gag/git/Genuine_Posed_EmotionRecognition/experiments/experiments_1st_data_split/experiment1/model.pth'

	# 1. Define the model class
	# Add the model class here
	
	loaded_model = Net()

	# 2. Load the model parameters
	# epoch = 10
	optimizer = torch.optim.SGD(loaded_model.parameters(), lr=0.01)
	# loss = nn.CrossEntropyLoss()


	loaded_checkpoint = torch.load(path)

	loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
	optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
	epoch = loaded_checkpoint['epoch']
	loss = loaded_checkpoint['loss']



	# for param in loaded_model.parameters():
	# 	print(param)
	# print(loaded_model.state_dict())

	# loaded_model.eval()

	# Get the data Local Paths
	train_dir = "data/train"
	test_dir = "data/test"

	
	dataset_train, dataset_test, dataset_classes = get_datasets()
	train_loader, test_loader = get_data_loaders(dataset_train, dataset_test)


	# Device configuration
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Test the model
	test_loss, test_accuracy = evaluate(loaded_model, test_loader, loss)
	print("Test Loss: {:.3f}.. ".format(test_loss),
		"Test Accuracy: {:.3f}".format(test_accuracy))

