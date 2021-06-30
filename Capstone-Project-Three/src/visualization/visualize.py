# visualize.py
import os
import pandas as pd
import seaborn as sns
import sys
sys.path.append('..')

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from  src import constants as con


def count_images(df):
    df_grouped = df.groupby('Morphology').count()
    df_grouped.rename(columns={'Image Dir': 'Image Count'}, inplace=True)
    df_sorted = df_grouped.sort_values(by='Image Count', ascending=False)
    return df_sorted


def display_images(image_list, num_cols):
    """
    Display a list of images from a list of image file names.

    :param image_list: List of image file names.
    :type image_list: list
    :param num_cols: The number of columns to use to arrange the images.
    :type num_cols: int
    """

    num_plots = len(image_list)
    num_rows = int(np.ceil(num_plots / num_cols))

    fig = plt.figure(figsize=(12, 12))
    for i, image_file in enumerate(image_list):
        img = cv.imread(os.path.join(con.RAW_IMAGES_DIR, image_file))
        fig.add_subplot(num_rows, num_cols, i+1)
        plt.title(image_file.split('/')[0])
        plt.axis('off')
        plt.imshow(img)
    plt.tight_layout()
    plt.show()


def display_images_by_channel(image_list):
    """
    Display images by channel.

    :param image_list: List of image file names.
    :type image_list: list
    """

    num_rows = len(image_list)
    img_tmp = cv.imread(os.path.join(con.RAW_IMAGES_DIR, image_list[0]))
    num_cols = img_tmp.shape[2]
    fig = plt.figure(figsize=(12, 48))
    counter = 1
    colors = ['Reds', 'Blues', 'Greens']
    for i, image_file in enumerate(image_list):
        img = cv.imread(os.path.join(con.RAW_IMAGES_DIR, image_file))
        for channel in range(num_cols):
            fig.add_subplot(num_rows, num_cols, counter)
            plt.title(f"{image_file.split('/')[0]} ({colors[channel][:-1].lower()})")
            plt.axis('off')
            plt.imshow(img[:, :, channel], cmap=colors[channel])
            counter += 1
    plt.tight_layout()
    plt.show()


def plot_maturity(df):
    """
    Create a donut plot to display image counts by maturity group.

    :params df: Table of distribution in image counts by maturity group.
    :type df: DataFrame
    """
    hole = plt.Circle((0, 0), 0.7, color='white')

    plt.figure(figsize=(8, 8))
    plt.pie(df['Count'].tolist(), labels=df.index.to_list(), colors=['red', 'green', 'blue'])
    p = plt.gcf()
    p.gca().add_artist(hole)
    plt.title('Distribution in Image Counts by Maturity Group')
    plt.show()


def compute_image_stats(df):
    """
    Compute statistics for each leukocyte image in the dataset.

    :params df: A dataset of image information.
    :type df: DataFrame
    :return: The computed image statistics.
    :rtype: DataFrame
    """
    print('Computing image statistics...')
    color = ['Red', 'Green', 'Blue']
    for image_file in df.index:
        img = cv.imread(os.path.join(con.RAW_IMAGES_DIR, image_file))
        for channel in range(img.shape[2]):
            df.loc[image_file, f'{color[channel]}_Max'] = np.max(img[:, :, channel])
            df.loc[image_file, f'{color[channel]}_Min'] = np.min(img[:, :, channel])
            df.loc[image_file, f'{color[channel]}_Mean'] = np.mean(img[:, :, channel])
            df.loc[image_file, f'{color[channel]}_Median'] = np.median(img[:, :, channel])
    print('Finished.')
    df.to_csv(os.path.join(con.PROCESSED_DATA_DIR, con.IMAGE_STATS))
    return df


def gather_image_stats(df):
    """
    Gather the image statistics. Either load the previously computed statistics, or calculate the statistics if they
    do not exist.

    :params df: A dataset of image information.
    :type df: DataFrame
    :return: The computed image statistics.
    :rtype: DataFrame
    """
    image_stats_file = os.path.join(con.PROCESSED_DATA_DIR, con.IMAGE_STATS)
    if os.path.exists(image_stats_file):
        df_image_stats = pd.read_csv(image_stats_file)
    else:
        df_image_stats = compute_image_stats(df)
    return df_image_stats


def plot_image_stats(df):
    """
    Create box plots for all image statistics. Each box plot is organized by image color channel and metric.

    :params df: A dataset of image information.
    :type df: DataFrame
    """

    channels = ['Red', 'Green', 'Blue']
    metrics = ['Max', 'Min', 'Mean', 'Median']

    num_plots = len(metrics) * len(channels)
    subplot_height = 6
    plot_height = subplot_height * num_plots
    fig, ax = plt.subplots(num_plots, 1, figsize=(18, plot_height))
    count = 0
    for metric in metrics:
        for i, channel in enumerate(channels):
            sns.boxplot(x='Morphology', y=f'{channel}_{metric}', data=df, color=channel.lower(), ax=ax[count])
            # Add transparency
            for patch in ax[count].artists:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b, 0.7))
            count += 1
    plt.show()