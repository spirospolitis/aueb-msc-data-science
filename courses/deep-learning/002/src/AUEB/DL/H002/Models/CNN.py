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
import pathlib
import tensorflow as tf

from .. import Env
from . import tf_callbacks

log = logging.getLogger("msc-ds-dl-h-002")

class CNN():
    def __init__(
        self, 
        config: {}
    ):
        self.__config = config
        self.__model = None
        self.__history = None

    def build(self):
        # Input layer (None, 28, 28)
        input_layer = tf.keras.layers.Input(
            shape = (self.__config["model"]["target_size"][0], self.__config["model"]["target_size"][1], self.__config["model"]["channels_size"]), 
            name = "input_layer"
        )
        
        l = input_layer

        # Stack convolutional layers.
        for i in range(1, self.__config["model"]["cnn"]["convolutional_layers"]["size"] + 1):
            
            # Kernel size (3, 3), (5, 5), (7, 7), ...
            if self.__config["model"]["cnn"]["convolutional_layers"]["kernel_growth_strategy"] == "funnel_out":
                kernel_size = ((i * 2) + 1, (i * 2) + 1)
            # Kernel size (11, 11), (9, 9), (7, 7), ...
            elif self.__config["model"]["cnn"]["convolutional_layers"]["kernel_growth_strategy"] == "funnel_in":
                kernel_size = (((self.__config["model"]["cnn"]["convolutional_layers"]["size"] + 1 - i) * 2) + 1, ((self.__config["model"]["cnn"]["convolutional_layers"]["size"] + 1 - i) * 2) + 1)
            elif self.__config["model"]["cnn"]["convolutional_layers"]["kernel_growth_strategy"] == "constant": 
                kernel_size = (3, 3)

            # Filters 32, 64, 128 ...
            if self.__config["model"]["cnn"]["convolutional_layers"]["filter_growth_strategy"] == "funnel_out":
                filters = np.power(2, i + 4)
            # Filters 128, 64, 32 ...
            elif self.__config["model"]["cnn"]["convolutional_layers"]["filter_growth_strategy"] == "funnel_in":
                filters = np.power(2, 8 - i)
            elif self.__config["model"]["cnn"]["convolutional_layers"]["filter_growth_strategy"] == "constant": 
                filters = 64

            l = tf.keras.layers.Conv2D(
                # Filters 32, 64, 128 ...
                # filters = np.power(2, i + 4), 
                filters = filters, 
                kernel_size = kernel_size, 
                strides = (1, 1), 
                padding = "same", 
                dilation_rate = (1, 1), 
                activation = "relu", 
                name = f"conv_2d_layer_{i}"
            )(l)

            # Apply batch normalization.
            #
            # Normalize the activations of the previous layer at each batch, i.e. applies a transformation 
            # that maintains the mean activation close to 0 and the activation standard deviation close to 1.
            # 
            # Source: https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
            if self.__config["model"]["cnn"]["batch_normalization"]["enabled"] == True:
                l = tf.keras.layers.BatchNormalization(
                    name = f"batch_normalization_layer_{i}"
                )(l)

            # Apply local pooling.
            # 
            # Downsamples the input representation by taking the maximum / average value over the window defined by pool_size 
            # for each dimension along the features axis. The window is shifted by strides in each dimension.
            # 
            # Source: https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D
            if self.__config["model"]["cnn"]["local_pooling"]["enabled"] == True:
                if self.__config["model"]["cnn"]["local_pooling"]["mode"] == "max":
                    l = tf.keras.layers.MaxPool2D(
                        pool_size = (2, 2), 
                        strides = (1, 1), 
                        padding = "same", 
                        name = f"max_pooling_2d_layer_{i}"
                    )(l)
                elif self.__config["model"]["cnn"]["local_pooling"]["mode"] == "avg":
                    l = tf.keras.layers.AveragePooling2D(
                        pool_size = (2, 2), 
                        strides = (1, 1), 
                        padding = "same", 
                        name = f"average_pooling_2d_layer_{i}"
                    )(l)
                else:
                    raise Exception("Invalid local pooling parameter")

            # Apply spatial dropout.
            #
            # Performs the same function as Dropout, however it drops entire 2D feature maps instead of individual elements.
            #
            # Source: https://www.tensorflow.org/api_docs/python/tf/keras/layers/SpatialDropout2D
            if self.__config["model"]["cnn"]["spatial_dropout"]["enabled"] == True:
                tf.keras.layers.SpatialDropout2D(
                    rate = self.__config["model"]["cnn"]["spatial_dropout"]["rate"], 
                    name = f"spatial_dropout_2d_layer_{i}"
                )

        # Apply global pooling.
        #
        # Global max / average pooling operation for spatial data.
        #
        # Source: https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalMaxPool2D
        if self.__config["model"]["cnn"]["global_pooling"]["enabled"] == True:
            if self.__config["model"]["cnn"]["global_pooling"]["mode"] == "max":
                l = tf.keras.layers.GlobalMaxPooling2D(
                    name = "global_max_pooling_2d_layer"
                )(l)
            elif self.__config["model"]["cnn"]["global_pooling"]["mode"] == "avg":
                l = tf.keras.layers.GlobalAveragePooling2D(
                    name = "global_average_pooling_2d_layer"
                )(l)
            else:
                raise Exception("Invalid global pooling parameter")

        # Flatten.
        l = tf.keras.layers.Flatten(
            name = "flatten_layer"
        )(l)
        
        # Fully connected layer dropout.
        if self.__config["model"]["cnn"]["fully_connected_dropout"]["enabled"] == True:
            l = tf.keras.layers.Dropout(
                rate = self.__config["model"]["cnn"]["fully_connected_dropout"]["rate"], 
                name = f"dropout_layer"
            )(l)

        # Output layer.
        output_layer = tf.keras.layers.Dense(
            units = 2, 
            activation = "softmax", 
            kernel_regularizer = self.__config["model"]["cnn"]["regularizer"]["kernel_regularizer"], 
            activity_regularizer = self.__config["model"]["cnn"]["regularizer"]["activity_regularizer"], 
            name = "output_layer"
        )(l)
        
        # Model definition.
        self.__model = tf.keras.models.Model(
            inputs = input_layer, 
            outputs = output_layer, 
            name = self.__config["model"]["model_instance_id"]
        )

        # Model compilation.
        self.__model.compile( 
            optimizer = self.__config["model"]["optimizer"], 
            loss = self.__config["model"]["loss"], 
            metrics = self.__config["model"]["metrics"]
        )

        # Plot the model architecture to a .png file.
        tf.keras.utils.plot_model(
            model = self.__model, 
            to_file = f"{pathlib.Path(os.path.join(pathlib.Path.cwd().parents[0], os.path.normpath(Env.OUTPUT_MODEL_BASE_DIR), os.path.normpath(Env.OUTPUT_MODEL_ARCHITECTURE_PLOTS_DIR), self.__config['model']['model_instance_id'] + '.png'))}", 
            show_shapes = True, 
            show_layer_names = True,
            dpi = 96
        )

        # log.info(f"{self.__model.summary()}")

        return self

    def fit(
        self, 
        train_data_frame_iterator, 
        validation_data_frame_iterator
    ):
        reduce_lr_on_plateau_cb = tf_callbacks.reduce_lr_on_plateau_callback(
            monitor = self.__config["training"]["reduce_lr_on_plateau"]["monitor"], 
            factor = self.__config["training"]["reduce_lr_on_plateau"]["factor"], 
            patience = self.__config["training"]["reduce_lr_on_plateau"]["patience"], 
            min_delta = self.__config["training"]["reduce_lr_on_plateau"]["min_delta"], 
            cooldown = self.__config["training"]["reduce_lr_on_plateau"]["cooldown"], 
            min_lr = self.__config["training"]["reduce_lr_on_plateau"]["min_lr"], 
            verbose = self.__config["training"]["reduce_lr_on_plateau"]["verbose"]
        )

        early_stopping_cb = tf_callbacks.early_stopping_callback(
            monitor = self.__config["training"]["early_stopping"]["monitor"], 
            patience = self.__config["training"]["early_stopping"]["patience"], 
            verbose = self.__config["training"]["early_stopping"]["verbose"]
        )

        checkpoint_cb = tf_callbacks.checkpoint_callback(
            filepath = f"{pathlib.Path(os.path.join(pathlib.Path.cwd().parents[0], os.path.normpath(Env.OUTPUT_MODEL_BASE_DIR), os.path.normpath(Env.OUTPUT_MODEL_CHECKPOINT_DIR), self.__config['model']['model_instance_id'] + '.hdf5'))}", 
            monitor = self.__config["training"]["checkpoint"]["monitor"], 
            mode = self.__config["training"]["checkpoint"]["mode"], 
            save_best_only = self.__config["training"]["checkpoint"]["save_best_only"], 
            verbose = self.__config["training"]["checkpoint"]["verbose"], 
        )

        cvs_logger_cb = tf_callbacks.csv_logger_callback(
            filename = f"{pathlib.Path(os.path.join(pathlib.Path.cwd().parents[0], os.path.normpath(Env.OUTPUT_MODEL_BASE_DIR), os.path.normpath(Env.OUTPUT_MODEL_LOGS_DIR), self.__config['model']['model_instance_id'] + '_training.csv'))}", 
            separator = "\t", 
            append = False
        )

        tensor_board_cb = tf_callbacks.tensor_board_callback(
            model_name = self.__config["model"]["model_instance_id"]
        )
            
        tqdm_cb = tf_callbacks.tqdm_callback()

        self.__history = self.__model.fit(
            x = train_data_frame_iterator, 
            steps_per_epoch = train_data_frame_iterator.n // train_data_frame_iterator.batch_size, 
            validation_data = validation_data_frame_iterator, 
            validation_steps = validation_data_frame_iterator.n // validation_data_frame_iterator.batch_size, 
            class_weight = self.__config["training"]["class_weights"], 
            epochs = self.__config["training"]["epochs"], 
            callbacks = [
                reduce_lr_on_plateau_cb, 
                early_stopping_cb, 
                checkpoint_cb, 
                cvs_logger_cb,  
                tqdm_cb
            ]
        )

        return self.__history

    @property
    def model(self):
        return self.__model

    @property
    def history(self):
        return self.__history