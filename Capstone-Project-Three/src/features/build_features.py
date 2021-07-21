# build_features.py
import os
import sys
sys.path.append('..')
from time import time

import cv2 as cv
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd

from src import constants as con


def preprocess_images(image_file_list, flatten=False, gray=False, rescale=False, scale_factor=1.0):
    preprocessed_images = []
    start = time()
    print('Preprocessing Images:')
    if flatten:
        print(' * Flattening.')
    if rescale:
        print(f' * Rescaling ({scale_factor:0.2%}).')
    if gray:
        print(' * Convert to grayscale.')
    for i, image_file in enumerate(image_file_list):
        image = cv.imread(os.path.join(con.RAW_IMAGES_DIR, image_file))
        if gray:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        if rescale:
            new_size = (int(image.shape[1]*scale_factor), int(image.shape[0]*scale_factor))
            image = cv.resize(image, new_size)
        if flatten:
            image = image.flatten()
        preprocessed_images.append(image.flatten())
    end = time()
    elapsed = end - start
    time_unit = 'seconds'
    if elapsed > 60:
        elapsed = elapsed / 60
        time_unit = 'minutes'
    print(f'It took {elapsed:0.3f} {time_unit} to load and preprocess {len(image_file_list)} images.')
    flattened_images_array = np.array(preprocessed_images)
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
    print(f'The array of preprocessed images takes up {memory_size:0.3f} {memory_unit} of memory.')
    return flattened_images_array


def bootstrap_images(X, y):
    """
    Boostrap the training data with replacement.

    :param X: Training data features.
    :type X: np.ndarray
    :param y: Training data labels.
    :type y: np.ndarray
    """

    # Find majority class
    class_fractions = pd.Series(y).value_counts(normalize=True)
    max_class_fraction = class_fractions.max()
    majority_class = class_fractions[class_fractions == max_class_fraction].index.values[0]
    df_train = pd.DataFrame(X, index=y)
    df_majority_class = df_train.loc[majority_class]

    # Upsample majority class by 10%
    majority_class_sample = df_majority_class.sample(frac=0.1, replace=True, random_state=42)

    # Combine with original training data
    df_upsampled_majority_class = pd.concat([df_train, majority_class_sample], axis=0)
    X_upsampled_majority_class = df_upsampled_majority_class.values
    y_upsampled_majority_class = df_upsampled_majority_class.index.values

    # Bootstrap training data to match size of upsampled majority class
    ros = RandomOverSampler(random_state=42)
    X_boostrapped, y_boostrapped = ros.fit_resample(X_upsampled_majority_class, y_upsampled_majority_class)

    return X_boostrapped, y_boostrapped
