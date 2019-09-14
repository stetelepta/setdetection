### Imports
import os
import sys
import numpy as np
import pandas as pd
import click
import logging
import pickle
import cv2
import resource
import random
from datetime import datetime
from pathlib import Path

from tensorflow.compat.v1 import set_random_seed
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression


def get_project_path():
    try:
        # running from a python file"
        project_path = Path(os.path.dirname(os.path.abspath(__file__)), os.pardir)
        is_file = True
        is_colab = False
    except NameError as e:
        # not running from a file (ie console or notebook)
        project_path = Path(os.getcwd(), os.pardir)
        is_file = False
        try:
            # running on google colab
            from google.colab import drive
            # connect to google drive
            drive.mount('/content/gdrive')
            project_path = Path("/content/gdrive/My Drive/Colab Notebooks/setdetection/")
            is_colab = True
        except ModuleNotFoundError:
            # not running on google colab
            is_colab = False
            output_path = project_path / "output"
    return project_path, is_file, is_colab


# get project path, and flag if script runs in a file
project_path, is_file, is_colab = get_project_path()
output_path = project_path / "output"
results_csv_path = output_path / 'results.csv'

# columns for results csv. This list is used to create the results dataframe
result_columns = ['run_id', 'nr_images', 'normalize', 'train_score', 'val_score']

# add project root to pythonpath
sys.path.insert(0, str(project_path / "src"))

# import custom packages
from utils.identify import *    
from utils.log import *
from utils.click_utils import conditionally_decorate
from set_cardgame.dataset import *


def get_or_create_results_file(results_csv_path, columns, sep=";"):
    try:
        df_results = pd.read_csv(results_csv_path, sep=sep)
    except FileNotFoundError:
        # create new results dataframe dictionary to store results
        df_results = pd.DataFrame(columns=columns)
    return df_results


# decorate with click commands if the script runs from a file
@conditionally_decorate(click.command(), is_file)
@conditionally_decorate(click.option('--normalize', default="none", help='how to normalize data, possible values: {"standard", "minimax"}.'),  is_file)
@conditionally_decorate(click.option('--nr_images', default=81, help='Number of images to generate for training'), is_file)
def train(normalize="none", nr_images=81):
    
    # create id for this run
    run_id = f'{normalize}_{nr_images}_{datetime.now().strftime("%d%m%Y-%H%M%S")}'
    
    # setup logging
    logger = setup_logger(level=logging.INFO, logfile=output_path / f"log_{run_id}.txt")

    # make results repeatable
    seed = 42
    random.seed(seed)  # `python` built-in pseudo-random generator
    np.random.seed(seed)  # numpy pseudo-random generator
    set_random_seed(seed)  # tensorflow pseudo-random generator
    logger.info(f"np.random.random(): {np.random.random()}")

    logger.info("======")
    logger.info(f"starting run {run_id}")
    logger.info(f"parameter: normalize: {normalize}")
    logger.info(f"parameter: nr_images: {nr_images}")
    logger.info(f"project_path: {project_path}")
    logger.info(f"output_path: {output_path}")
    logger.info(f"is_file: {is_file}")
    logger.info(f"is_colab: {is_colab}")

    # load training and validation data
    X_train, y_train, X_val, y_val = load_dataset(nr_images=nr_images, output_path=None)

    # nr samples
    m_train = X_train.shape[0]
    m_val = X_val.shape[0]
    
    logger.debug("X_train.shape:", X_train.shape)
    logger.debug("X_val.shape:", X_val.shape)
    logger.debug("m_train:", m_train)
    logger.debug("m_val:", m_val)

    # get results dataframe, or if it is not there create it
    df_results = get_or_create_results_file(results_csv_path, result_columns)

    # keep track of memory usage
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    logger.info("===")

    # log memory usage for each iteration
    mem_usage1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logger.info(f"Memory usage: {mem_usage1}, diff: {mem_usage1-mem_usage}")
    mem_usage = mem_usage1

    # reshape embeddings to 2D
    X_train = X_train.reshape(m_train, -1)
    X_val = X_val.reshape(m_val, -1)
    
    if normalize == "standard":
        logger.info("normalize using standard scaling")
        scaler = StandardScaler()
    elif normalize == "minmax":
        logger.info("normalize using standard scaling")
        scaler = MinMaxScaler()
    else:
        logger.info("no normalization")
        scaler = None

    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    # fit model
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)

    train_score = clf.score(X_train, y_train)
    val_score = clf.score(X_val, y_val)
    logger.info(f"score for baseline, nr_images: {nr_images}, train_score: {train_score:.4f}, val_score: {val_score:.4f}")
    
    # store results to csv
    row = {}
    row['run_id'] = run_id
    row['nr_images'] = nr_images
    row['normalize'] = normalize
    row['train_score'] = train_score
    row['val_score'] = val_score
    df_row = pd.DataFrame(data=[row])
    df_results = pd.concat([df_results, df_row], ignore_index=True, sort=False)
    df_results.to_csv(results_csv_path, sep=";", index=False)


if __name__ == '__main__':
    if is_file:
        # provide arguments from command line
        train()
    else:
        # provide arguments in the code
        train(normalize="none", nr_images=810)
