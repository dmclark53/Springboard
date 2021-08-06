# evaluate_model.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y, y_pred, label_encodings):

    y_flat = np.argmax(y, axis=1)
    y_pred_flat = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_flat, y_pred_flat)
    df_cm = pd.DataFrame(cm, index=label_encodings.keys(), columns=label_encodings.keys())

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()
