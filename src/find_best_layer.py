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
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

# import supported models
from tensorflow.keras.applications import ResNet50, MobileNet, MobileNetV2

# import preprocessing functions for the models
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_input_mobile_net
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobile_net_v2

from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam
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
result_columns = ['run_id', 'layer_id', 'layer_name', 'model', 'nr_images', 'normalize', 'train_score', 'val_score']

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
@conditionally_decorate(click.option('--model', default="resnet", help='which model to use for embeddings: {"resnet", "mobilenet", "mobilenet_v2"}.'),  is_file)
def train(normalize="none", nr_images=81, model="resnet"):
    
    # create id for this run
    run_id = f'{model}_{normalize}_{nr_images}_{datetime.now().strftime("%d%m%Y-%H%M%S")}'
    
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
    logger.info(f"parameter: model: {model}")
    logger.info(f"project_path: {project_path}")
    logger.info(f"output_path: {output_path}")
    logger.info(f"is_file: {is_file}")
    logger.info(f"is_colab: {is_colab}")

    # initialize intermediate model
    if model == "resnet":
        model_instance = ResNet50(include_top=False, weights='imagenet', pooling=None, input_shape=(96, 128, 3))
        preprocess_input_function = preprocess_input_resnet50
    elif model == "mobilenet":
        model_instance = MobileNet(include_top=False, weights='imagenet', pooling=None, input_shape=(96, 128, 3))
        preprocess_input_function = preprocess_input_mobile_net
    elif model == "mobilenet_v2":
        model_instance = MobileNetV2(include_top=False, weights='imagenet', pooling=None, input_shape=(96, 128, 3))
        preprocess_input_function = preprocess_input_mobile_net_v2
    else:
        logger.error(f"unsupported model: {model}")
        return

    # load training and validation data
    X_train, y_train, X_val, y_val = load_dataset(nr_images=nr_images, output_path=None)

    # apply preprocessing for the specified model
    X_train = preprocess_input_function(X_train)
    X_val = preprocess_input_function(X_val)

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

    # loop randomly through layers
    nr_layers = len(model_instance.layers)
    random_index = np.random.choice(range(1, nr_layers), size=nr_layers-1, replace=False)
    for i, layer_id in enumerate(random_index):
        logger.info("===")
        logger.info(f"process layer: {i} - {layer_id}")
        # log memory usage for each iteration
        mem_usage1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        logger.info(f"Memory usage: {mem_usage1}, diff: {mem_usage1-mem_usage}")
        mem_usage = mem_usage1

        layer_name = model_instance.layers[layer_id].name

        # check if there is a result for this layer_id, nr_images, normalize combination
        cond_layer_id = (df_results['layer_id'] == layer_id)
        cond_model = (df_results['model'] == model)
        cond_nr_images = (df_results['nr_images'] == nr_images)
        cond_normalize = (df_results['normalize'] == normalize)
        df_previous_results = df_results[cond_layer_id & cond_model & cond_nr_images & cond_normalize]
        logger.info(f"{len(df_previous_results)} previous results for model {model}, layer {layer_id}, nr_images {nr_images}, and normalize settings: {normalize}")
        if len(df_previous_results) > 0:
            logger.info(f"skip iteration - found previous results for model {model}, layer {layer_id}, nr_images {nr_images}, and normalize settings: {normalize}.")
            logger.debug(f"records found: {df_previous_results}")
            continue

        # compile model, use specific layer as output
        intermediate_model = Model(inputs=model_instance.input, outputs=model_instance.get_layer(layer_name).output)
        intermediate_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=[categorical_accuracy])

        # generate image embeddings
        X_train_embeddings = intermediate_model.predict(X_train)
        X_val_embeddings = intermediate_model.predict(X_val)

        # reshape embeddings to 2D
        X_train_embeddings = X_train_embeddings.reshape(m_train, -1)
        X_val_embeddings = X_val_embeddings.reshape(m_val, -1)
        
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
            X_train_embeddings = scaler.fit_transform(X_train_embeddings)
            X_val_embeddings = scaler.transform(X_val_embeddings)

        # fit model
        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train_embeddings, y_train)

        train_score = clf.score(X_train_embeddings, y_train)
        val_score = clf.score(X_val_embeddings, y_val)
        logger.info(f"score for layer {layer_id} - layer_name {layer_name}, train_score: {train_score:.4f}, val_score: {val_score:.4f}")
        
        # store results to csv
        row = {}
        row['run_id'] = run_id
        row['layer_id'] = layer_id
        row['layer_name'] = layer_name
        row['model'] = model
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
        train(normalize="none", nr_images=810, model="mobilenet")
