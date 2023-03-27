# Multi-Label, Multi-Task Deep Learning Approach Towards Detection The Differences Between Real And Fake Emotions


## Repository structure

```
data -> contains data for the Single-task learning approach (src-stl) and the original dataset.
data_mtl -> contains data for the Multi-task learning approach (src-mtl).
data_seq -> contains data for the temporal approach (src-temp).
experiments -> experiments of all approaches.
src-stl -> Single-task learning approach.
src-mtl -> Multi-task learning approach.
src-temp -> Temporal approach.
utils -> data analysis and statistics from the dataset.
```

## Requirements for this project
```
pip install -r requirements.txt
```

## src-stl
________________________________________________________________________________
| File name | Description |
| --- | --- |
| data.py | data utils related to the dataset, (dataloades, transformations, etc) |
| dataset_prep.py | python file used for the creation of the dataset (frames) |
| evaluate.py | python file for evaluation of models (need to define the path of the model) |
| fine_tune.py | fine tuning of a selected model, freezes the baseline mode and adds classification head |
| loss_functions.py | loss functions used in this approach |
| models.py | models classes in this approach |
| train.py | training and validation function |
| utils.py | utilities function of this approach |
| main.py | main python file to run (training + validation) |

## src-mtl
________________________________________________________________________________
| File name | Description |
| --- | --- |
| confusion_matrix.py | generates confussion matrix |
| dataset_prep.py | python file used for the creation of the MTL dataset  |
| evaluate.py | python file for evaluation of models (need to define the path of the model) |
| models.py | models classes in this approach |
| predictions_mtl.py | file that predicts classes on a given input image |
| single_prediction.py | file that makes predictions in a single image |
| train.py | training and validation function |
| utils.py | utilities function of this approach |
| main.py | main python file to run (training + validation) |

## src-temp
________________________________________________________________________________
| File name | Description |
| --- | --- |
| frames.py | generate frames from each video |
| sequences.py | generate the sequences of frames |
| models.py | models classes in this approach |
| dataset.py | dataset class |
| utils.py | utilities function of this approach |
| main.py | main python file to run (training + validation + evaluation) |
| main_notebook.ipynb | Full code for training and eval model of temporal approach in jupyter notebook (used in google colab) |


## Instructions
* After installing the requirements for this project, you can access the data (find the link for data in data/README.md) and experiments via google drive (for experiments in experiments/README.md).

### Train models:
* You can train the models by running the main.py file in each approach. For example, to train the model for the single-task learning approach you need to run the following command:

```python src-stl/main.py```

* For the multi-task learning approach you need to run the following command:

```python src-mtl/main.py```

* For the temporal approach you need to run the following command, or the main_notebook.py.
```python src-temp/main.py```



### Test models:
* For testing models you need to define the path of the model in the evaluate.py file for both src-stl and src-mtl. For src-temp the main file can be used also for evaluation. The trained models can be found in experiments/README.md.



---------------------
### References
Bendjoudi, I., Vanderhaegen, F., Hamad, D., & Dornaika, F. (2021). Multi-label, multi-task CNN approach for context-based emotion recognition. Information Fusion, 76, 422–428. https://doi.org/10.1016/j.inffus.2020.11.007

Dahling, J., & Perez, L. (2010). Older worker, different actor? Linking age and emotional labor strategies. Personality and Individual Differences - PERS INDIV DIFFER, 48, 574–578. https://doi.org/10.1016/j.paid.2009.12.009

Frédéric, V., & Zieba, S. (2014). Reinforced learning systems based on merged and cumulative knowledge to predict human actions. Information Sciences, 276, 146–159. https://doi.org/10.1016/j.ins.2014.02.051

Gross, J. J., & Levenson, R. W. (1995). Emotion elicitation using films. Cognition and Emotion, 9(1), 87–108. https://doi.org/10.1080/02699939508408966

Isaacowitz, D. M. (2012). Mood Regulation in Real Time: Age Differences in the Role of Looking. Current Directions in Psychological Science, 21(4), 237–242. https://doi.org/10.1177/0963721412448651

Johnston, L., Miles, L., & Macrae, C. N. (2010). Why are you smiling at me? Social functions of enjoyment and non- enjoyment smiles. British Journal of Social Psychology, 49(1), 107–127. https://doi.org/10.1348/014466609X412476

Kim, Y.-G., & Huynh, X.-P. (2017). Discrimination Between Genuine Versus Fake Emotion Using Long-Short Term Memory with Parametric Bias and Facial Landmarks. 2017 IEEE International Conference on Computer Vision Workshops (ICCVW), 3065–3072. https://doi.org/10.1109/ICCVW.2017.362

Kulkarni, K., Corneanu, C. A., Ofodile, I., Escalera, S., Baro, X., Hyniewska, S., Allik, J., & Anbarjafari, G. (2018). Automatic Recognition of Facial Displays of Unfelt Emotions (arXiv:1707.04061). arXiv. http://arxiv.org/abs/1707.04061

Ready, R. E., Santorelli, G. D., & Mather, M. A. (2017). Judgment and classification of emotion terms by older and younger adults. Aging & Mental Health, 21(7), 684–692. https://doi.org/10.1080/13607863.2016.1150415

Saxen, F., Werner, P., & Al-Hamadi, A. (2017, October 1). Real vs. Fake Emotion Challenge: Learning to Rank Authenticity From Facial Activity Descriptors. https://doi.org/10.1109/ICCVW.2017.363
