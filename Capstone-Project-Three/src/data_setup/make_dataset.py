# make_dataset.py
import os
import pickle

from src import constants as con


def load_train_test():
    """
    Load pickled training and test data from file.
    """

    file_list = ['X_train', 'X_test', 'y_train', 'y_test']
    data_sets = []
    for filename in file_list:
        data_sets.append(pickle.load(open(os.path.join(con.PROCESSED_DATA_DIR, filename), 'rb')))
    return tuple(data_sets)
