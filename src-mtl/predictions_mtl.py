import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import ChimeraNet
from dataset import SASEFE_MTL_TEST
from tqdm.auto import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def visualize_model(model, prediction_text_fn, num_images=6):
	was_training = model.training
	model.eval()
	images_so_far = 0
	fig = plt.figure()
	
	counter = 0

	with torch.no_grad():
		for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)): 
			counter += 1
			inputs = data["image"].to(device)

			real_fake_label = data["real_fake"].to(device)
			emotion_label = data["emotion"].to(device)
			
			real_fake_output, emotion_output = model(inputs)


			# for j in range(inputs.size()[0]):
			# 	images_so_far += 1 
			# 	ax = plt.subplot(num_images//2, 2, images_so_far)
			# 	ax.axis('off')
			# 	ax.set_title('predicted: {}'.format(prediction_text_fn(real_fake_output, j)))
			# 	plt.imshow(inputs.cpu().data[j])

			# 	if images_so_far == num_images:
			# 		model.train(mode=was_training)
			# 		return
				
			# for j in range(inputs.size()[0]):
			# 	images_so_far += 1 
			# 	ax = plt.subplot(num_images//2, 2, images_so_far)
			# 	ax.axis('off')
			# 	ax.set_title('predicted: {}'.format(prediction_text_fn(emotion_output, j)))
			# 	plt.imshow(inputs.cpu().data[j])
			# 	plt.show()

			# 	if images_so_far == num_images:
			# 		model.train(mode=was_training)
			# 		return

		model.train(mode=was_training)




def multi_prediction_text_fn(emotion_output,real_fake_output,idx):
	_, preds_real_fake = torch.max(real_fake_output.data, 1)
	_, preds_emotion = torch.max(emotion_output.data, 1)
	return '{}, {}'.format(preds_real_fake.item(), preds_emotion.item())




if __name__ == '__main__':
	# Load the model.pth - change the path to the model you want to evaluate
    path = 'experiments/exp1-chimeranet/model.pth'

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define the model class
    loaded_model = ChimeraNet().to(device)

    # Define the optimizer
    optimizer = torch.optim.SGD(loaded_model.parameters(), lr=1e-4, momentum=0.09)

    # Define the loss functions.
    emotion_loss = nn.CrossEntropyLoss()
    real_fake_loss = nn.CrossEntropyLoss()

    # Load the checkpoint
    loaded_checkpoint = torch.load(path, map_location=device)
    loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    epoch = loaded_checkpoint['epoch']
    
	# Load the data
    test_dir = "data_mtl/test"
    test_image_paths = os.listdir(test_dir)

    test_dataset = SASEFE_MTL_TEST(test_image_paths)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
	# Visualize the model

    visualize_model(loaded_model, multi_prediction_text_fn)
    
    
    
    
