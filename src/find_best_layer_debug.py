### Imports
import os
import sys
import numpy as np
import pandas as pd
import click
import logging
import pickle
import cv2
import gc
import resource
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# path settings
try:
    # Running from a python file"
    project_path = Path(os.path.dirname(os.path.abspath(__file__)), os.pardir)
except NameError as e:
    # Running from console or notebook"
    project_path = Path(os.getcwd(), os.pardir)
output_path = project_path / "output"

# add project root to pythonpath
sys.path.insert(0, str(project_path / "src"))

# import custom packages
from utils.identify import *    
from utils.log import *
from set_cardgame.dataset import *

# setup logging
logger = setup_logger(level=logging.INFO)


def layer_output_generator(gen, model, layer_name, input_shape):
    # use model generate features for a specific layer
    intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=[categorical_accuracy])
     
    for x, y in gen:
        yield intermediate_model.predict(x), y


@click.command()
@click.option('--normalize', default=None, help='how to normalize data, possible values: {"standard", "minimax"}.')
@click.option('--nr_images', default=81, help='Number of images to generate for training')
def train(normalize, nr_images):
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logger.info('Memory usage: %s (kb)' % mem_usage)

    logger.info(f"parameter: normalize: {normalize}")
    logger.info(f"parameter: nr_images: {nr_images}")

    # load training and validation data
    X_train, y_train, X_val, y_val = load_dataset(nr_images=nr_images, output_path=None)

    logger.debug("X_train.shape:", X_train.shape)
    logger.debug("X_val.shape:", X_val.shape)

    # load meta data
    df_meta, df_meta_dummies = load_metadata()

    # initialize intermediate model
    resnet50 = ResNet50(include_top=False, weights='imagenet', pooling=None, input_shape=(96, 128, 3))

    # resnet preprocessing
    X_train = preprocess_input_resnet50(X_train)
    X_val = preprocess_input_resnet50(X_val)

    # nr samples
    m_train = X_train.shape[0]
    m_val = X_val.shape[0]

    # dictionary to store results
    results = {
        'layer_id': [],
        'layer_name': [], 
        'nr_images': [], 
        'normalize': [],
        'train_score': [], 
        'val_score': []
    }


    # loop randomly through layers
    nr_layers = len(resnet50.layers)
    random_index = np.random.choice(nr_layers, size=nr_layers, replace=False)
    
    # test with model outside loop    
    layer_name = resnet50.layers[100].name
    logger.info(f"{layer_name}")

    # compile model, use specific layer as output
    intermediate_model = Model(inputs=resnet50.input, outputs=resnet50.get_layer(layer_name).output)
    intermediate_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=[categorical_accuracy])
       
    lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial') 

    for i, layer_id in enumerate(random_index):    
        mem_usage1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        logger.info('Memory usage: %s (kb)' % (mem_usage1))
        logger.info('Memory usage diff: %s (kb)' % (mem_usage1-mem_usage))
        mem_usage = mem_usage1

        # csv file for storing results for this layer
        layer_output_csv = output_path / f'results{layer_id}.csv'
        
        # # check if there is a result for this layer (allows for subsequent execution) 
        # if layer_output_csv.is_file():
        #     logger.warning(f"skipping layer: {layer_id}, found previous results for this layer")
        #     continue
            
        # # generate image embeddings
        X_train_embeddings = intermediate_model.predict(X_train)
        #X_val_embeddings = intermediate_model.predict(X_val)

        # # reshape embeddings to 2D
        X_train_embeddings = X_train_embeddings.reshape(m_train, -1)
        # X_val_embeddings = X_val_embeddings.reshape(m_val, -1)
        
        # if normalize == "standard":
        #     logger.info("normalize using standard scaling")
        #     scaler = StandardScaler(copy=False)
        # elif normalize == "minmax":
        #     logger.info("normalize using standard scaling")
        #     scaler = MinMaxScaler(copy=False)
        # else:
        #     logger.info("no normalization")
        #     scaler = None

        # if scaler:
        #     logger.info("normalize data")
        #     X_train_embeddings = scaler.fit_transform(X_train_embeddings)
        #     #X_val_embeddings = scaler.transform(X_val_embeddings)

        # fit model
        clf = lr.fit(X_train_embeddings, y_train)
        del clf
        del X_train_embeddings
        #train_score = clf.score(X_train_embeddings, y_train)
        # val_score = clf.score(X_val_embeddings, y_val)
        # logger.info(f"score for layer {layer_id} - layer_name {layer_name}, train_score: {train_score:.2f}, val_score: {val_score:.2f}")
        
        # # store results
        # results['layer_id'].append(layer_id)
        # results['layer_name'].append(layer_name)
        # results['nr_images'].append(nr_images)
        # results['normalize'].append(normalize)
        # results['train_score'].append(train_score)
        # results['val_score'].append(-1)
        
        # # export results to csv
        # results_df = pd.DataFrame(data=results)
        # results_df.to_csv(layer_output_csv)

        # # garbage collection
        # gc.collect()


if __name__ == '__main__':
    train()
