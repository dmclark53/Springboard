# visualize.py
import os
import sys

import numpy as np

sys.path.append('..')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from  src import constants as con


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
        img = mpimg.imread(os.path.join(con.RAW_IMAGES_DIR, image_file))
        fig.add_subplot(num_rows, num_cols, i+1)
        plt.title(image_file.split('/')[0])
        plt.imshow(img)
    plt.tight_layout()
    plt.show()
