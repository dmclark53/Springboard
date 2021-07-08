# build_features.py
import os
import sys
sys.path.append('..')
from time import time

import cv2 as cv
import numpy as np

from src import constants as con


def flatten_images(image_file_list):
    flattened_images = []
    start = time()
    for image_file in image_file_list:
        image = cv.imread(os.path.join(con.RAW_IMAGES_DIR, image_file))
        flattened_images.append(image.flatten)
    end = time()
    elapsed = end - start
    time_unit = 'seconds'
    if elapsed > 60:
        elapsed = elapsed / 60
        time_unit = 'minutes'
    print(f'It took {elapsed:0.3f} {time_unit} to load and flatten {len(image_file_list)} images.')
    flattened_images_array = np.array(flattened_images)
    memory_size = flattened_images_array.size * flattened_images_array.itemsize
    print(f'The array of flattened images takes up {int(memory_size)} bytes of memory.')
    return flattened_images_array
