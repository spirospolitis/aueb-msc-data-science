"""
    AUEB M.Sc. in Data Science
    Semester: Sprint 2020
    Course: Deep Learning
    Homework: 2
    Lecturer: P. Malakasiotis
    Author: Spiros Politis
"""

import os
import tensorflow as tf
import tensorflow_addons as tfa

# If no change on the 'monitor' parameter is achieved, ReduceLROnPlateau 
# will reduce the learning rate which is often beneficial.
def reduce_lr_on_plateau_callback(
    monitor: str = "val_loss", 
    factor: float = 0.1, 
    patience: int = 10, 
    min_delta: float = 0.0001, 
    cooldown: int = 0, 
    min_lr: float = 0, 
    verbose: int = 0
):
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor = monitor, 
        factor = factor, 
        patience = patience, 
        mode = "auto", 
        min_delta = min_delta, 
        cooldown = cooldown, 
        min_lr = min_lr, 
        verbose = verbose 
    )
    
def early_stopping_callback(
    monitor: str = "val_loss", 
    min_delta: int = 0, 
    patience: int = 0, 
    mode: str = "auto", 
    verbose: int = 0
):
    return tf.keras.callbacks.EarlyStopping(
        monitor = monitor, 
        min_delta = min_delta, 
        patience = patience, 
        mode = mode, 
        restore_best_weights = True, 
        verbose = verbose
    )

def checkpoint_callback(
    filepath: str = None, 
    monitor: str = "val_acc", 
    mode: str = "auto", 
    save_best_only: bool = True, 
    verbose: int = 0
): 
    return tf.keras.callbacks.ModelCheckpoint(
        filepath = filepath, 
        monitor = monitor, 
        mode = mode, 
        save_best_only = save_best_only, 
        verbose = verbose
    )

def csv_logger_callback(
    filename: str = "training_results.csv", 
    separator: str = "\t", 
    append: bool = False
):
    return tf.keras.callbacks.CSVLogger(
        filename = filename, 
        separator = separator, 
        append = append
    )

def tensor_board_callback(model_name:str = None):     
    return tf.keras.callbacks.TensorBoard(
        log_dir = f"logs{os.path.sep}{model_name}", 
        histogram_freq = 0, 
        write_graph = True, 
        write_images = False,
        update_freq = "epoch", 
        profile_batch = 2, 
        embeddings_freq = 0, 
        embeddings_metadata = None
    )

def tqdm_callback():
    return tfa.callbacks.TQDMProgressBar()