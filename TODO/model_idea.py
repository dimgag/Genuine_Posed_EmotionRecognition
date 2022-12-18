# Code template for the function to predict the emotion
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Defining the VGGFace model
class VGGFace(nn.Module):
    def __init__(self):
        super(VGGFace, self).__init__()
        self.features = models.vgg16(pretrained=True).features
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2622),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Loading the model
model = VGGFace()
model.load_state_dict(torch.load('data/SASE-FE/vgg_face_dag.pth'))

# Defining the preprocessing function
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])

# Defining the postprocessing function
from scipy.spatial.distance import cosine

def postprocess(output):
    output = output.squeeze(0)
    output = F.normalize(output, p=2, dim=0)
    return output

# Defining the function to predict the emotion
def predict_emotion(image):
    # Preprocess the image
    image_tensor = preprocess(image)
    # Add an extra batch dimension since pytorch treats all images as batches
    image_tensor.unsqueeze_(0)
    # Turn the input into a Variable
    input = torch.autograd.Variable(image_tensor)
    # Predict the emotion
    output = model(input)
    # Postprocess the output
    output = postprocess(output)
    return output

# Predicting the emotion
output = predict_emotion(img_zoom)
print(output.shape)

# Defining the function to find the most similar emotion
def find_most_similar_emotion(embedding):
    # Load the emotions embeddings
    emotions = torch.load('data/SASE-FE/emotions.pt')
    # Find the most similar emotion
    min_distance = 1
    for emotion in emotions:
        distance = cosine(emotion, embedding)
        if distance < min_distance:
            min_distance = distance
            most_similar_emotion = emotion
    return most_similar_emotion

# Finding the most similar emotion
most_similar_emotion = find_most_similar_emotion(output)
print(most_similar_emotion.shape)

# Defining the function to find the emotion label
def find_emotion_label(embedding):
    # Load the emotions embeddings
    emotions = torch.load('data/SASE-FE/emotions.pt')
    # Load the emotions labels
    emotions_labels = torch.load('data/SASE-FE/emotions_labels.pt')
    # Find the most similar emotion
    min_distance = 1
    for i, emotion in enumerate(emotions):
        distance = cosine(emotion, embedding)
        if distance < min_distance:
            min_distance = distance
            most_similar_emotion = emotion
            most_similar_emotion_label = emotions_labels[i]
    return most_similar_emotion_label

# Finding the emotion label
most_similar_emotion_label = find_emotion_label(output)
print(most_similar_emotion_label)

# Plotting the image and the predicted emotion
plt.imshow(img_zoom)
plt.title(most_similar_emotion_label)
plt.axis("off")
plt.show()




