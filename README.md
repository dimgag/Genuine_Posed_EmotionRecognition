# Genuine Posed EmotionRecognition
![ViewCount](https://views.whatilearened.today/views/github/dimgag/Genuine-Posed-EmotionRecognition.svg)
![GitHub repo size](https://img.shields.io/github/repo-size/dimgag/Genuine-Posed-EmotionRecognition)
![GitHub last commit](https://img.shields.io/github/last-commit/dimgag/Genuine-Posed-EmotionRecognition)
![GitHub forks](https://img.shields.io/github/forks/dimgag/Genuine-Posed-EmotionRecognition?style=social)
![GitHub stars](https://img.shields.io/github/stars/dimgag/Genuine-Posed-EmotionRecognition?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/dimgag/Genuine-Posed-EmotionRecognition?style=social)

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction
### Master in Data Science for Decision Making
Student - Dimitrios Gagatsis
Thesis Supervisor - Mirela Popa
This is a reopository of my master thesis that is about to start.
The Thesis name is "Multimodal approach for Detection of Genuine and Posed Facial Expressions of Emotions"
The Data sources for this project are: SASE-FE Database and K-EmoCon Dataset 

## 🚀 Models
<!-- Generate table -->

| Model | Paper |
|  ---  |  ---  |
| VGG-LSTM | [paper](https://www.researchgate.net/publication/339836787_Pedestrian_Navigation_Method_Based_on_Machine_Learning_and_Gait_Feature_Assistance) | 
| ResNet50-LSTM |[paper](https://www.hindawi.com/journals/wcmc/2020/8909458/) |
| SENet-LSTM | [paper](https://ieeexplore.ieee.org/document/9568952) |
| 3D-CNN | [paper](https://keras.io/examples/vision/3D_image_classification/#:~:text=A%203D%20CNN%20is%20simply,learning%20representations%20for%20volumetric%20data.) |
| ResNet3D | [paper](https://paperswithcode.com/model/resnet-3d?variant=resnet-3d-18) |


## Approach
```mermaid
graph TD
    Input --> Classifier_Model_1
	Classifier_Model_1 --> Anger
	Classifier_Model_1 --> Disgust
	Classifier_Model_1 --> Fear
	Classifier_Model_1 --> Happiness
	Classifier_Model_1 --> Sadness
	Classifier_Model_1 --> Surprise
	Anger --> Classifier_Model_2
	Disgust --> Classifier_Model_2
	Fear --> Classifier_Model_2
	Happiness --> Classifier_Model_2
	Sadness --> Classifier_Model_2
	Surprise --> Classifier_Model_2
	Classifier_Model_2 --> Genuine_emotion
	Classifier_Model_2 --> Posed_emotion
```

