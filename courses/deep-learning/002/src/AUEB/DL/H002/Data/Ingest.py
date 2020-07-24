"""
    AUEB M.Sc. in Data Science
    Semester: Sprint 2020
    Course: Deep Learning
    Homework: 2
    Lecturer: P. Malakasiotis
    Author: Spiros Politis
"""

import logging
import numpy as np
import os
import pandas as pd
import pathlib
import re
import sklearn.model_selection
import tensorflow as tf

from .. import Env

from . import preprocess_crop

CLASSES_BINARY = [
    "0", 
    "1"
]

CLASSES_MULTI_CLASS = [
    "XR_ELBOW", 
    "XR_FINGER", 
    "XR_FOREARM", 
    "XR_HAND", 
    "XR_HUMERUS", 
    "XR_SHOULDER", 
    "XR_WRIST"
]

# Set random seed on Numpy and TensorFlow.
np.random.seed(Env.RANDOM_SEED)
tf.random.set_seed(Env.RANDOM_SEED)

log = logging.getLogger("msc-ds-dl-h-002")

"""
"""
def train_val_binary(
    batch_size: int = 32, 
    target_size: tuple = (256, 256), 
    shuffle: bool = False, 
    
    # Data augmentation parameters.
    rotation_range: int = None, 
    horizontal_flip: bool = False, 
    vertical_flip: bool = False, 
    
    # Validation split parameters.
    validation_split: int = None, 
    stratified_split: bool = True, 

    filter_classes: [] = None
):
    log.info(\
        f"Ingesting train set for binary classification task with parameters: \n \
        \t|__ batch_size = {batch_size}, \n \
        \t|__ target_size = {target_size}, \n \
        \t|__ shuffle = {shuffle}, \n \
        \t|__ rotation_range = {rotation_range}, \n \
        \t|__ horizontal_flip = {horizontal_flip}, \n \
        \t|__ vertical_flip = {vertical_flip}, \n \
        \t|__ validation_split = {validation_split}" \
    )

    # Ingest train_labeled_studies.csv.
    labeled_studies_df = pd.read_csv(
        os.path.normpath(os.path.join(pathlib.Path.cwd().parents[0], Env.DATA_SOURCE_BASE_DIR, "train_labeled_studies.csv")), 
        header = None, 
        names = ["path", "class"], 
        dtype = {
            "path": str, 
            "class": str
        }
    )
    labeled_studies_df["path"] = labeled_studies_df["path"].astype(str)
    labeled_studies_df["class"] = labeled_studies_df["class"].astype(str)

    # Ingest train_image_paths.csv.
    image_paths_df = pd.read_csv(
        os.path.normpath(os.path.join(pathlib.Path.cwd().parents[0], Env.DATA_SOURCE_BASE_DIR, "train_image_paths.csv")), 
        header = None,
        names = ["path"], 
        dtype = {
            "path": str
        }
    )
    image_paths_df["path"] = image_paths_df["path"].astype(str)
    image_paths_df["full_path"] = image_paths_df["path"]
    image_paths_df["path"] = image_paths_df["path"].apply(lambda x: re.sub(r"image*.*", "", x))

    # Join DataFrames.
    df = labeled_studies_df.merge(image_paths_df, on = "path", how = "inner") \
        .drop(columns = ["path"]) \
        .rename(columns = {"full_path": "path"})
    
    if filter_classes is not None:
        df = df[df["path"].str.contains('|'.join(filter_classes))]

    # Get image data generator from DataFrame.
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        # Scale pixels values to [0-1].
        rescale = 1.0 / 255.0, 
        rotation_range = rotation_range, 
        horizontal_flip = horizontal_flip, 
        vertical_flip = vertical_flip, 
    )
    
    if validation_split is not None and validation_split != 0:
        
        if stratified_split == True:
            # Stratified train / validation split.
            train_df, validation_df = sklearn.model_selection.train_test_split(df, test_size = validation_split, stratify = df["class"])
        else:
            # Random train / test split.
            train_df, validation_df = sklearn.model_selection.train_test_split(df, test_size = validation_split)

        train_generator = image_data_generator.flow_from_dataframe(
            train_df, 
            directory = "../data", 
            x_col = "path", 
            y_col = "class", 
            weight_col = None, 
            target_size = target_size, 
            color_mode = "rgb", 
            classes = CLASSES_BINARY, 
            class_mode = "categorical", 
            batch_size = batch_size, 
            shuffle = shuffle, 
            seed = Env.RANDOM_SEED, 
            save_to_dir = None, 
            save_prefix = "processed_train", 
            save_format = "png", 
            # interpolation = "nearest", 
            interpolation = "lanczos:center",
            validate_filenames = True, 
            subset = None
        )

        validation_generator = image_data_generator.flow_from_dataframe(
            validation_df, 
            directory = "../data", 
            x_col = "path", 
            y_col = "class", 
            weight_col = None, 
            target_size = target_size, 
            color_mode = "rgb", 
            classes = CLASSES_BINARY, 
            class_mode = "categorical", 
            batch_size = batch_size, 
            shuffle = shuffle, 
            seed = Env.RANDOM_SEED, 
            save_to_dir = os.path.normpath(os.path.join(pathlib.Path.cwd().parents[0], Env.DATA_SOURCE_BASE_DIR, Env.DATA_PROCESSED_DIR)), 
            save_prefix = "processed_valid", 
            save_format = "png", 
            # interpolation = "nearest", 
            interpolation = "lanczos:center", 
            validate_filenames = True, 
            subset = None
        )

        return train_generator, validation_generator

    else:

        train_generator = image_data_generator.flow_from_dataframe(
            df, 
            directory = "../data", 
            x_col = "path", 
            y_col = "class", 
            weight_col = None, 
            target_size = target_size, 
            color_mode = "rgb", 
            classes = CLASSES_BINARY, 
            class_mode = "categorical", 
            batch_size = batch_size, 
            shuffle = shuffle, 
            seed = RANDOM_SEED, 
            save_to_dir = os.path.normpath(os.path.join(pathlib.Path.cwd().parents[0], Env.DATA_SOURCE_BASE_DIR, Env.DATA_PROCESSED_DIR)), 
            save_prefix = "processed_train", 
            save_format = "png", 
            # interpolation = "nearest", 
            interpolation = "lanczos:center", 
            validate_filenames = True, 
            subset = None
        )

        return train_generator, None
        
"""
"""
def test_binary(
    batch_size: int = 32, 
    target_size: tuple = (256, 256), 
    shuffle: bool = False, 
    filter_classes: [] = None
):
    # log.debug("Ingesting test dataset for binary classification task")

    # Ingest valid_labeled_studies.csv.
    labeled_studies_df = pd.read_csv(
        os.path.normpath(os.path.join(pathlib.Path.cwd().parents[0], Env.DATA_SOURCE_BASE_DIR, "valid_labeled_studies.csv")), 
        header = None,
        names = ["path", "class"], 
        dtype = {
            "path": str, 
            "class": str
        }
    )
    labeled_studies_df["path"] = labeled_studies_df["path"].astype(str)

    # Ingest valid_image_paths.csv.
    image_paths_df = pd.read_csv(
        os.path.normpath(os.path.join(pathlib.Path.cwd().parents[0], Env.DATA_SOURCE_BASE_DIR, "valid_image_paths.csv")), 
        header = None,
        names = ["path"], 
        dtype = {
            "path": str
        }
    )
    image_paths_df["path"] = image_paths_df["path"].astype(str)
    image_paths_df["full_path"] = image_paths_df["path"]
    image_paths_df["path"] = image_paths_df["path"].apply(lambda x: re.sub(r"image*.*", "", x))

    # Join DataFrames.
    df = labeled_studies_df.merge(image_paths_df, on = "path", how = "inner") \
        .drop(columns = ["path"]) \
        .rename(columns = {"full_path": "path"})
    
    if filter_classes is not None:
        df = df[df["path"].str.contains('|'.join(filter_classes))]

    # Get image data generator from DataFrame.
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1.0 / 255.0
    ).flow_from_dataframe(
        df, 
        directory = "../data", 
        x_col = "path", 
        y_col = "class", 
        weight_col = None, 
        target_size = target_size, 
        color_mode = "rgb", 
        classes = CLASSES_BINARY, 
        class_mode = "categorical", 
        batch_size = batch_size, 
        shuffle = shuffle, 
        seed = Env.RANDOM_SEED, 
        save_to_dir = os.path.normpath(os.path.join(pathlib.Path.cwd().parents[0], Env.DATA_SOURCE_BASE_DIR, Env.DATA_PROCESSED_DIR)), 
        save_prefix = "processed", 
        save_format = "png", 
        subset = None, 
        # interpolation = "nearest", 
        interpolation = "lanczos:center",
        validate_filenames = True
    )

    return image_data_generator