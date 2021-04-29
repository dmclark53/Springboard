# viz.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_ecdfs(df, columns_list, num_cols, exclude_zeros=False, figure_name=None):

    num_plots = len(columns_list)
    num_rows = num_plots // num_cols

    fix, ax = plt.subplots(num_rows, num_cols, figsize=(12, 4*num_rows))
    for i, column in enumerate(columns_list):
        row_i = i // num_cols
        col_i = i % num_cols
        if len(ax.shape) == 2:
            next_ax = ax[row_i, col_i]
        else:
            next_ax = ax[i]
        if exclude_zeros:
            non_zero = df[column] != 0.0
            sns.ecdfplot(x=column, data=df[non_zero], ax=next_ax)
        else:
            sns.ecdfplot(x=column, data=df, ax=next_ax)
        next_ax.set_title(f'ECDF for {column}')
    plt.tight_layout()
    if figure_name is not None:
        plt.savefig(f'../reports/images/{figure_name}.png')
    else:
        pass
    plt.show()


def plot_kdes(df, columns_list, num_cols, exclude_zeros=False, save_figure=False):

    num_plots = len(columns_list)
    num_rows = num_plots // num_cols

    fix, ax = plt.subplots(num_rows, num_cols, figsize=(16, 4*num_rows))
    for i, column in enumerate(columns_list):
        row_i = i // num_cols
        col_i = i % num_cols
        if len(ax.shape) == 2:
            next_ax = ax[row_i, col_i]
        else:
            next_ax = ax[i]
        if exclude_zeros:
            non_zero = df[column] != 0.0
            sns.kdeplot(x=column, hue='status_group', data=df[non_zero], ax=next_ax)
        else:
            sns.kdeplot(x=column, hue='status_group', data=df, ax=next_ax)
    plt.tight_layout()
    if save_figure:
        plt.savefig('../reports/images/kde_plots.png')
    plt.show()


def plot_cat_hists(df, cat_counts, max_count=None, min_count=0, response_var='status_group', save_figure=False):

    if max_count is None:
        max_count = cat_counts.max()

    columns_to_plot = cat_counts[(cat_counts >= min_count) & (cat_counts < max_count)].index.values
    columns_to_plot = np.delete(columns_to_plot, np.where(columns_to_plot == response_var))
    num_rows = len(columns_to_plot)

    sns.set(font_scale=1.5)
    fix, ax = plt.subplots(len(columns_to_plot), 1, figsize=(16, 3 * num_rows))
    for i, column in enumerate(columns_to_plot):
        df.loc[:, column] = df.loc[:, column].cat.as_ordered()
        # print(f'{column}:')
        # print(f'{df.loc[:, column].unique()}')
        sns.histplot(x=column, data=df, hue=response_var, multiple='stack', ax=ax[i])
    plt.tight_layout()
    if save_figure:
        plt.savefig('../reports/images/cat_hist_plots.png')
    plt.show()

