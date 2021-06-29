# visualize.py
import os
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
    : type num_cols: int
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
