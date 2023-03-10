'''
MAIN IDEA:

I want to implement a Multi-Task Learning Transformer model with temporal information for Image Classification.
The model should be able to predict the following tasks:
Classify the emotion of a person in a given video
Classify if the emotion is real or fake

For that purpose I have the following dataset:
data_temporal /
  train_root/
           /fake_angry
           /fake_contempt
           /fake_disgust
           /fake_happy
           /fake_sad
           /fake_surprise
           /real_angry
           /real_contempt
           /real_disgust
           /real_happy
           /real_sad
           /real_surprise
  val_root/
           /fake_angry
           /fake_contempt
           /fake_disgust
           /fake_happy
           /fake_sad
           /fake_surprise
           /real_angry
           /real_contempt
           /real_disgust
           /real_happy
           /real_sad
           /real_surprise
Where each folder contains videos.

'''