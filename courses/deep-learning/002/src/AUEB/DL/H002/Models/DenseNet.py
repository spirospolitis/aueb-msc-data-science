"""
    AUEB M.Sc. in Data Science
    Semester: Sprint 2020
    Course: Deep Learning
    Homework: 2
    Lecturer: P. Malakasiotis
    Author: Spiros Politis
"""

import logging
import os
import pathlib
import tensorflow as tf

from .. import Env
from . import tf_callbacks

log = logging.getLogger("msc-ds-dl-h-002")

"""
    Transfer learning with the DenseNet architecture.
    
    Densely Connected Convolutional Networks [1] allow for increased depth of deep convolutional networks
    by eliminating the vanishing gradients problem.

    [1] G. Huang, Z. Liu and L. van der Maaten, "Densely Connected Convolutional Networks", 2018.
"""
class DenseNet():
    def __init__(
        self, 
        config: {}
    ):
        self.__config = config
        self.__model = None
        self.__history = None

    def build(self):
        trainable_layers = 0

        if self.__config["model"]["model_id"] == "DenseNet121":
            dense_net_layer = tf.keras.applications.DenseNet121(
                input_shape = (self.__config["model"]["target_size"][0], self.__config["model"]["target_size"][1], self.__config["model"]["channels_size"]), 
                include_top = False, 
                weights = self.__config["model"]["dense_net"]["weights"], 
                pooling = self.__config["model"]["dense_net"]["pooling"]
            )

        if self.__config["model"]["model_id"] == "DenseNet169":
            dense_net_layer = tf.keras.applications.DenseNet169(
                input_shape = (self.__config["model"]["target_size"][0], self.__config["model"]["target_size"][1], self.__config["model"]["channels_size"]), 
                include_top = False, 
                weights = self.__config["model"]["dense_net"]["weights"], 
                pooling = self.__config["model"]["dense_net"]["pooling"]
            )

        if self.__config["model"]["model_id"] == "DenseNet201":
            dense_net_layer = tf.keras.applications.DenseNet201(
                input_shape = (self.__config["model"]["target_size"][0], self.__config["model"]["target_size"][1], self.__config["model"]["channels_size"]), 
                include_top = False, 
                weights = self.__config["model"]["dense_net"]["weights"], 
                pooling = self.__config["model"]["dense_net"]["pooling"], 
            )

        l = dense_net_layer.output

        if self.__config["head"] == "deep_mlp":
            # l = tf.keras.layers.GlobalAveragePooling2D(
            #     name = "global_average_pooling_layer_1"
            # )(l)
            # trainable_layers += 1

            # Batch normalization normalizes the output of a previous activation layer 
            # by subtracting the batch mean and dividing by the batch standard deviation, 
            # thereby keeping activation values within limits.
            l = tf.keras.layers.BatchNormalization(
                name = "batch_normalization_layer_1"
            )(l)
            trainable_layers += 1
            
            # l = tf.keras.layers.Dropout(
            #     rate = 0.2, 
            #     name = "dropout_layer_1"
            # )(l)
            # trainable_layers += 1

            l = tf.keras.layers.Dense(
                units = 1024, 
                activation = "relu", 
                name = "dense_layer_1"
            )(l)
            trainable_layers += 1

            l = tf.keras.layers.Dropout(
                rate = 0.5, 
                name = "dropout_layer_2"
            )(l)
            trainable_layers += 1

            l = tf.keras.layers.Dense(
                units = 512, 
                activation = "relu", 
                name = "dense_layer_2"
            )(l) 
            trainable_layers += 1

            l = tf.keras.layers.Dropout(
                rate = 0.2, 
                name = "dropout_layer_3"
            )(l)
            trainable_layers += 1

            l = tf.keras.layers.BatchNormalization(
                name = "batch_normalization_layer_2"
            )(l)
            trainable_layers += 1
            
        output_layer = tf.keras.layers.Dense(
            units = self.__config["model"]["classes_size"], 
            activation = "sigmoid", 
            name = "output_layer"
        )(l)
        trainable_layers += 1

        self.__model = tf.keras.Model(
            inputs = dense_net_layer.input, 
            outputs = output_layer, 
            name = self.__config["model"]["model_instance_id"]
        )

        # Avoid training of DenseNet layers.
        # Beneficial for two reasons:
        #
        # 1) Avoid overfitting the DenseNet.
        # 2) Trainable parameters are kept in check.

        for layer in self.__model.layers[0:-trainable_layers]:
            layer.trainable = False

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
            mode = self.__config["training"]["early_stopping"]["mode"], 
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
