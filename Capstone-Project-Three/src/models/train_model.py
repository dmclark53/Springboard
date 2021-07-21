# train_model
import os
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src import constants as con


def run_model(model, X_train, y_train, X_test, y_test):
    # Find model type
    model_type = str(type(model)).split("'")[1].split('.')[-1]
    print(f'model type: {model_type}')

    # Create model directory if it doesn't exist
    model_directory = os.path.join(con.MODELS_DIR, model_type)
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Run model
    start = time()
    model.fit(X_train, y_train)
    end = time()
    elapsed = end - start
    time_unit = 'seconds'
    if elapsed > 3600:
        elapsed = elapsed / 3600
        time_unit = 'hours'
    elif elapsed > 60:
        elapsed = elapsed / 60
        time_unit = 'minutes'
    else:
        pass

    print(f'It took the model {elapsed:0.3f} {time_unit} to run.')

    # Gather parameters
    df_params = pd.DataFrame(model.get_params(), index=[0])
    df_params['num_features'] = X_train.shape[1]
    df_params['num_rows'] = X_train.shape[0]
    df_params['elapsed_time'] = end - start

    # Create parameters file if it doesn't exist, otherwise append to it
    params_file = os.path.join(model_directory, 'params.csv')
    model_num = 0
    if os.path.exists(params_file):
        df_params_old = pd.read_csv(params_file)
        df_concat = pd.concat([df_params_old, df_params], axis=0)
        df_concat.index = np.arange(len(df_concat))
        df_concat.to_csv(params_file, index=False)
        model_num = len(df_concat)
    else:
        df_params.to_csv(params_file, index=False)
        model_num = 1

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)

    # Save classification report for training data
    cr_train = classification_report(y_train, y_pred_train, output_dict=True)
    df_cr_train = pd.DataFrame(cr_train).T
    cr_train_file = f'classification_report_train_{model_num}.csv'
    df_cr_train.to_csv(os.path.join(model_directory, cr_train_file))

    # Save classification report for testing data
    cr_test = classification_report(y_test, y_pred, output_dict=True)
    df_cr_test = pd.DataFrame(cr_test).T
    cr_test_file = f'classification_report_test_{model_num}.csv'
    df_cr_test.to_csv(os.path.join(model_directory, cr_test_file))
