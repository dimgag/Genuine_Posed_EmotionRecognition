# Crerate Confusion Matrix class

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ConfusionMatrix:
    def __init__(self, y_true, y_pred, labels=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels

    def get_confusion_matrix(self):
        return pd.crosstab(self.y_true, self.y_pred, rownames=['Actual'], colnames=['Predicted'])

    def plot_confusion_matrix(self, normalize=False):
        cm = self.get_confusion_matrix()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = cm.round(2)
            title = 'Normalized Confusion Matrix'
        else:
            title = 'Confusion Matrix'
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()


