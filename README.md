# Multi-Label, Multi-Task Deep Learning Approach Towards Detection The Differences Between Real And Fake Emotions
## Table of Contents
- [Repository Structure](#repository-structure)
- [💾 Dataset](#dataset)
- [🚀 Approaches](#approaches)
- [📈 Model Results](#model-results)
- [References](#references)


---------------------
## Repository Structure
```bash
├── README.md
├── LICENSE
├── data
│   ├── raw
│   ├── processed
│   └── external
├── models
│   ├── model1
│   ├── model2
│   └── model3
├── notebooks
│   ├── notebook1.ipynb
│   ├── notebook2.ipynb
│   └── notebook3.ipynb
├── src
│   ├── data
│   │   ├── make_dataset.py
│   │   └── process_dataset.py
│   ├── features
│   │   ├── build_features.py
│   │   └── feature_selection.py
│   ├── models
│   │   ├── train_model.py
│   │   └── predict_model.py
│   ├── visualization
│   │   ├── visualize.py
│   │   └── report.py
│   └── utils
│       ├── __init__.py
│       └── helpers.py
├── reports
│   ├── figures
│   └── tables
├── requirements.txt
└── setup.py
```
---------------------
## 💾 Dataset:
The SASE-FE Dataset (Kulkarni et al., 2018) created by the iCV Research Lab contains 643 different videos captured with high-resolution cameras recording 100 frames per second, containing video recordings of 54 participants of ages 19-36. The main reason behind the choice of such an age-range sample is that older adults have different, more positive responses than younger adults about feelings and emotions, and they are faster and more precise to regulate emotional states than younger adults (Dahling & Perez, 2010; Isaacowitz, 2012; Ready et al., 2017). More specifically, for each recording, participants were asked to act two facial expressions of emotions in a sequence, a genuine and a posed emotion. Genuine emotions are the six expressions happiness, sadness, anger, disgust, contempt, and surprise. For eliciting genuine and realistic emotions, proposed videos based on emotion science research (Gross & Levenson, 1995), were shown to the participants to increase the realism of their emotions. To increase the distinction between the two facial expressions presented in the sequence, the two emotions were chosen based on their visual and conceptual differences. Thus, the contrast was created by asking the participants to act happy after being Sad, Surprised after being Sad, Disgusted after being Happy, Sad after being Happy, Angry after being Happy, and Contemptuous after being Happy. Note that the participants were asked to start their video recordings from a neutral face and none of the participants were aware of the fact that they would be asked to act with a second facial expression.

Dataset Labels Mapping:
```JSON
    mapping = {
        "D2N2Sur.MP4": "fake_surprise",
        "H2N2A.MP4": "fake_angry",
        "H2N2C.MP4": "fake_contempt",
        "H2N2D.MP4": "fake_disgust",
        "H2N2S.MP4": "fake_sad",
        "S2N2H.MP4": "fake_happy",
        "N2A.MP4": "real_angry",
        "N2C.MP4": "real_contempt",
        "N2D.MP4": "real_disgust",
        "N2H.MP4": "real_happy",
        "N2S.MP4": "real_sad",
        "N2Sur.MP4": "real_surprise"
    }
```

---------------------
## 🚀 Approaches:
### 1. Spatial Video Classification (frames - 12 labels)

### 2. Multi-Task Learning Approach (frames - 6xEmotions + 2xReal/Fake)

### 3. Temporal Video Classification (video - 12 labels)


---------------------
## 📈 Model Results
### 1. Spatial Video Classification (frames - 12 labels)
| Model | Accuracy | Loss |
| :---: | :---: | :---: |
| ... | ... | ... |


### 2. Multi-Task Learning Approach (frames - 6xEmotions + 2xReal/Fake)
| Model | Accuracy | Loss |
| :---: | :---: | :---: |
| ... | ... | ... |


### 3. Temporal Video Classification (video - 12 labels)
| Model | Accuracy | Loss |
| :---: | :---: | :---: |
| ... | ... | ... |




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
