# Show prediction on a single image
# Usage: python single_prediction.py -img <image_path>
import os
import torch
import torch.nn as nn
from torchvision.transforms.functional import resize, to_tensor, normalize
from models import ChimeraNet
from PIL import Image
import cv2
import argparse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Predict real/fake and emotion on a single image')
parser.add_argument('-img', '--image_path', type=str, default='Image.jpeg', help='path to image')
args = parser.parse_args()




def predict_single_image(model, image_path):
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize, normalize, and convert to tensor
    image = resize(image, (224, 224))
    image = normalize(to_tensor(image), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    
    # Add batch dimension
    image = image.unsqueeze(0)
    
    # Move to device
    image = image.to(device)
    
    model.eval()
    # Forward pass
    real_fake_output, emotion_output = model(image)
    _, preds_real_fake = torch.max(real_fake_output.data, 1)
    _, preds_emotion = torch.max(emotion_output.data, 1)

    
    # Return predicted labels
    return preds_real_fake.item(), preds_emotion.item()


if __name__ == '__main__':
    # Load the model.pth - change the path to the model you want to evaluate
    path = 'experiments/experiments_MTL/exp1-chimeranet/model.pth'

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

    # Make a prediction on a single image
    # image_path = '/Users/dim__gag/Desktop/Image.jpeg'

    # image_path = '/Users/dim__gag/Desktop/Image2.jpeg'
    image_path = args.image_path
    
    real_fake_label, emotion_label = predict_single_image(loaded_model, image_path)
    
    # Print the predicted labels
    print("Real/Fake label: {}, Emotion label: {}".format(real_fake_label, emotion_label))

    # Add the names to the predicted labels
    if real_fake_label == 0:
        real_fake_label = 'Fake'
    else:
        real_fake_label = 'Real'

    if emotion_label == 0:
        emotion_label = 'Happy'
    elif emotion_label == 1:
        emotion_label = 'Sad'
    elif emotion_label == 2:
        emotion_label = 'Surprise'
    elif emotion_label == 3:
        emotion_label = 'Disgust'
    elif emotion_label == 4:
        emotion_label = 'Contempt'
    else:
        emotion_label = 'Angry'


    # plot the image and the predicted labels
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    cv2.putText(image, "Real/Fake: {}".format(real_fake_label), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, "Emotion: {}".format(emotion_label), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)


    





