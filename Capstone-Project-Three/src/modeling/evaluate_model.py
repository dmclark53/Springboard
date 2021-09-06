# evaluate_model.py
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src import constants as con


def plot_confusion_matrix(y, y_pred, label_encodings, figure_name=None):

    y_flat = np.argmax(y, axis=1)
    y_pred_flat = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_flat, y_pred_flat)
    df_cm = pd.DataFrame(cm, index=label_encodings.keys(), columns=label_encodings.keys())

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    if figure_name is not None:
        plt.rcParams['font.size'] = '16'
        plt.savefig(os.path.join(con.FIGURES_DIR, f'{figure_name}.png'))
    else:
        pass
    plt.show()


def plot_train_val_losses(results, figure_name=None):

    df_history = pd.DataFrame(results.history)

    max_epoch = df_history.index.values.max()
    if max_epoch > 12:
        tick_sep = max_epoch // 6
    else:
        tick_sep = 1
    xtick_values = np.arange(0, max_epoch, tick_sep)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.lineplot(data=df_history.loc[:, ['loss', 'val_loss']], ax=ax)
    ax.set_title('Loss by Epoch')
    ax.set_xticks(xtick_values)
    ax.set_xticklabels(xtick_values)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    if figure_name is not None:
        plt.rcParams['font.size'] = '16'
        plt.savefig(os.path.join(con.FIGURES_DIR, f'{figure_name}.png'))
    else:
        pass
    plt.show()
