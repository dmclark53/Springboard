# viz.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_ecdfs(df, columns_list, num_cols, exclude_zeros=False):

    num_plots = len(columns_list)
    num_rows = num_plots // num_cols

    fix, ax = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    for i, column in enumerate(columns_list):
        row_i = i // num_cols
        col_i = i % num_cols
        if exclude_zeros:
            non_zero = df[column] != 0.0
            sns.ecdfplot(x=column, data=df[non_zero], ax=ax[row_i, col_i])
        else:
            sns.ecdfplot(x=column, data=df, ax=ax[row_i, col_i])
    plt.tight_layout()
    plt.show()


def plot_kdes(df, columns_list, num_cols, exclude_zeros=False):

    num_plots = len(columns_list)
    num_rows = num_plots // num_cols

    fix, ax = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    for i, column in enumerate(columns_list):
        row_i = i // num_cols
        col_i = i % num_cols
        if exclude_zeros:
            non_zero = df[column] != 0.0
            sns.kdeplot(x=column, hue='status_group', data=df[non_zero], ax=ax[row_i, col_i])
        else:
            sns.kdeplot(x=column, hue='status_group', data=df, ax=ax[row_i, col_i])
    plt.tight_layout()
    plt.show()
