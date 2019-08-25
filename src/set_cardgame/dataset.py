'''
code for preparing and loading dataset for the SET game
'''

### imports
import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# dataset path
dataset_path = Path(os.path.dirname(os.path.abspath(__file__))) / "data" / "128x96"
csv_path = dataset_path.parent / "dataset.csv"

# setup logger
logger = logging.getLogger(__name__)


def get_labels_from_iterator(directory_iterator, y):
    '''
    return array with class labels from a generator. Helper function to get labels as convenient np array

    Args: 
        directory_iterator (DirectoryIterator instance): The Iterator instance returned by flow_from_directory
        y (nd array):  Should be a onehot vector of shape (m, 81)
    '''

    # class_indices contains a dictionary mapping class indices to classnames
    # we use this to obtain a classnames vector, where the index corresponds to the 
    # class index, and the value to the classname
    class_names = np.array(list(directory_iterator.class_indices.items()))[:, 0]

    # return classnames for y
    return class_names[np.argmax(y, axis=1)]


def upsample_dataset(images_path, nr_images, aug_params={}, target_size=(96, 128), batch_size=81, shuffle=True, seed=42, output_path=None):
    '''
    return generated images and target labels as numpy arrays
    
    Args:
        images_path (Pathlib.path): Path that contains the dataset (should contain subfolders for each label)
        nr_images (integer): Amount of images to return
        aug_params (dict): Parameters for ImageDataGenerator. Use if you want to use augmentation.
        target_size (tuple of integers): The dimensions (height, width) to which all images found will be resized. (default: (96, 128)
        batch_size (integer): Size of the batches of data. (default 81)
        shuffle (boolean): Whether to shuffle the data (default: True)
        seed (integer): Random seed for shuffling and transformations. (default 42)
    '''
    
    flow_params = {
        'target_size': target_size,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'seed': seed 
    }

    if output_path:
        flow_params.update({'save_to_dir': str(output_path / "gen")})

    # initalize generator and augment all images in dataset folder
    directory_iterator = ImageDataGenerator(**aug_params).flow_from_directory(images_path, **flow_params)
    
    # nr of batches we need to reach nr_images, using batch_size of 81
    nr_batches = np.ceil(nr_images / 81).astype(int)
    
    X = np.array([]) # matrix with resulting shape: (nr_images, height, width, 3)  
    y = np.array([]) # vector with resulting shape: (nr_images, 81)
    
    # get all batches and concatenate them 
    for batch in np.arange(nr_batches):
        X_batch, y_batch = next(directory_iterator)
        if len(X) == 0:
            X = X_batch
            y = y_batch
        else:
            X = np.concatenate([X, X_batch])
            y = np.concatenate([y, y_batch])

    return X[0:nr_images], get_labels_from_iterator(directory_iterator, y[0:nr_images])


def load_metadata():
    # dataframe with info on each card and variant
    df_meta = pd.read_csv(csv_path)
    # create dataframe with dummie variables for each card
    df_meta_card = df_meta.drop_duplicates(subset='card_id')
    df_meta_card.set_index('card_id', inplace=True)
    df_meta_card = df_meta_card.drop(columns=['filename', 'variant'])

    # create dummie variables
    df_meta_dummies = pd.get_dummies(df_meta_card)
    return df_meta, df_meta_dummies
    

def load_dataset(target_size=(96, 128), nr_images=810, shuffle=True, output_path=None):
    
    # augmentation parameters for training data
    aug_params = {
        'shear_range': 0.4,
        'zoom_range': 0.4,
        'rotation_range': 45,
        'horizontal_flip': True,
        'vertical_flip': True
    }

    # get training data
    X_train, y_train = upsample_dataset(dataset_path / "train", nr_images=nr_images, aug_params=aug_params, target_size=target_size, batch_size=81, shuffle=shuffle, output_path=output_path)


    # get validation data
    X_val, y_val = upsample_dataset(dataset_path / "validation", nr_images=81, aug_params={}, target_size=target_size, batch_size=81, shuffle=shuffle, output_path=None)
    

    return X_train, y_train, X_val, y_val