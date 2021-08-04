# make_dataset.py
import os
import pickle

from src import constants as con


def load_train_test(data_dir):
    """
    Load pickled training and test data from file.
    """

    data_directories = [os.path.split(x[0])[-1] for x in os.walk(con.PROCESSED_DATA_DIR)]
    if data_dir in data_directories:
        file_list = ['X_train', 'X_test', 'y_train', 'y_test']
        data_sets = []
        for filename in file_list:
            data_sets.append(pickle.load(open(os.path.join(con.PROCESSED_DATA_DIR, data_dir, filename), 'rb')))
        return tuple(data_sets)
    else:
        print(f'The data directory {data_dir} not in {con.PROCESSED_DATA_DIR}.')
        return None, None, None, None
