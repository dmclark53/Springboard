# build_features.py
import os
import sys
sys.path.append('..')
from time import time

import cv2 as cv
import numpy as np

from src import constants as con


def flatten_images(image_file_list, rescale=False):
    flattened_images = []
    start = time()
    for image_file in image_file_list:
        image = cv.imread(os.path.join(con.RAW_IMAGES_DIR, image_file))
        if rescale:
            scale = 0.5
            new_size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
            image = cv.resize(image, new_size)
        flattened_images.append(image.flatten())
    end = time()
    elapsed = end - start
    time_unit = 'seconds'
    if elapsed > 60:
        elapsed = elapsed / 60
        time_unit = 'minutes'
    print(f'It took {elapsed:0.3f} {time_unit} to load and flatten {len(image_file_list)} images.')
    flattened_images_array = np.array(flattened_images)
    memory_size = flattened_images_array.size * flattened_images_array.itemsize
    memory_unit = 'bytes'
    if memory_size >= 1e9:
        memory_size = memory_size / 1e9
        memory_unit = 'Gb'
    elif memory_size >= 1e6:
        memory_size = memory_size / 1e6
        memory_unit = 'Mb'
    elif memory_size >= 1e3:
        memory_size = memory_size / 1e3
        memory_unit = 'Kb'
    else:
        pass
    print(f'The array of flattened images takes up {memory_size:0.3f} {memory_unit} of memory.')
    return flattened_images_array
