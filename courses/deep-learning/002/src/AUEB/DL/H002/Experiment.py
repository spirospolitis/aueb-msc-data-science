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
import sklearn

from . import Env
from .Data import Ingest
from .Models import CNN, DenseNet
from .Visualization import plot_training_history

log = logging.getLogger("msc-ds-dl-h-002")

class Experiment():
    def __init__(
        self, 
        model_id: str, 
        batch_size: int = 32, 
        target_size: () = (224, 244), 
        channels_size: int = 3, 
        classes_size: int = 2, 
        shuffle: bool = True, 
        trainable: bool = False, 
        validation_split: float = 0.1, 
        data_augmentation: bool = False, 
        filter_classes: [] = None, 
        model_config: {} = None, 
        random_seed: int = None
    ):
        self.__model_id = model_id
        self.__batch_size = batch_size
        self.__target_size = target_size
        self.__channels_size = channels_size
        self.__classes_size = classes_size 
        self.__trainable = trainable
        self.__data_augmentation = data_augmentation
        self.__validation_split = validation_split
        self.__shuffle = shuffle
        self.__random_seed = random_seed
        self.__filter_classes = filter_classes

        self.__model_config = model_config
        
        self.__experiment_params = None

        self.__histories = None

    def __build_experiment_params(self):
        self.__experiment_params = []
        
        log.debug(f"Experiment::__build_experiment_params:: class filter is {self.__filter_classes}")

        if self.__filter_classes is None:
            if self.__data_augmentation == False:
                experiment_params = {
                    "model_id": self.__model_id, 
                    "batch_size": self.__batch_size, 
                    "target_size": self.__target_size, 
                    "channels_size": self.__channels_size, 
                    "classes_size": self.__classes_size, 
                    "trainable": self.__trainable, 
                    "data_augmentation": self.__data_augmentation, 
                    "horizontal_flip": False, 
                    "vertical_flip": False, 
                    "rotation_range": 0, 
                    "filter_classes": self.__filter_classes, 
                    "validation_split": self.__validation_split, 
                    "shuffle": self.__shuffle, 
                    "random_seed": self.__random_seed
                }
            else:
                experiment_params = {
                    "model_id": self.__model_id, 
                    "batch_size": self.__batch_size, 
                    "target_size": self.__target_size, 
                    "channels_size": self.__channels_size, 
                    "classes_size": self.__classes_size, 
                    "trainable": self.__trainable, 
                    "data_augmentation": self.__data_augmentation, 
                    "horizontal_flip": True, 
                    "vertical_flip": True, 
                    "rotation_range": 30, 
                    "filter_classes": self.__filter_classes, 
                    "validation_split": self.__validation_split, 
                    "shuffle": self.__shuffle, 
                    "random_seed": self.__random_seed
                }
            
            self.__experiment_params.append(experiment_params)
        else:
            for filter_class in self.__filter_classes:
                if self.__data_augmentation == False:
                    experiment_params = {
                        "model_id": self.__model_id, 
                        "batch_size": self.__batch_size, 
                        "target_size": self.__target_size, 
                        "channels_size": self.__channels_size, 
                        "classes_size": self.__classes_size, 
                        "trainable": self.__trainable, 
                        "data_augmentation": self.__data_augmentation, 
                        "horizontal_flip": False, 
                        "vertical_flip": False, 
                        "rotation_range": 0, 
                        "filter_classes": [filter_class], 
                        "validation_split": self.__validation_split, 
                        "shuffle": self.__shuffle, 
                        "random_seed": self.__random_seed
                    }
                else:
                    experiment_params = {
                        "model_id": self.__model_id, 
                        "batch_size": self.__batch_size, 
                        "target_size": self.__target_size, 
                        "channels_size": self.__channels_size, 
                        "classes_size": self.__classes_size, 
                        "trainable": self.__trainable, 
                        "data_augmentation": self.__data_augmentation, 
                        "horizontal_flip": True, 
                        "vertical_flip": True, 
                        "rotation_range": 30, 
                        "filter_classes": [filter_class], 
                        "validation_split": self.__validation_split, 
                        "shuffle": self.__shuffle, 
                        "random_seed": self.__random_seed
                    }
                
                self.__experiment_params.append(experiment_params)
                
        return self

    def run(self):
        self.__build_experiment_params()
        self.__histories = {}

        for experiment_params in self.__experiment_params:

            train_data_frame_iterator, validation_data_frame_iterator = Ingest.train_val_binary(
                batch_size = experiment_params["batch_size"], 
                target_size = experiment_params["target_size"], 
                shuffle = experiment_params["shuffle"], 
                rotation_range = experiment_params["rotation_range"], 
                horizontal_flip = experiment_params["horizontal_flip"], 
                vertical_flip = experiment_params["vertical_flip"], 
                validation_split = experiment_params["validation_split"], 
                filter_classes = experiment_params["filter_classes"]
            )

            test_data_frame_iterator = Ingest.test_binary(
                batch_size = experiment_params["batch_size"], 
                target_size = experiment_params["target_size"], 
                shuffle = experiment_params["shuffle"], 
                filter_classes = experiment_params["filter_classes"]
            )

            class_weights = sklearn.utils.class_weight.compute_class_weight(
                class_weight = "balanced", 
                classes = np.unique(train_data_frame_iterator.classes), 
                y = train_data_frame_iterator.classes
            )
            class_weights = {i : class_weights[i] for i in range(experiment_params["classes_size"])}
            
            if "CNN" in self.__model_id:
                if self.__filter_classes is None:
                    model_instance_id = f"{experiment_params['model_id']}" \
                        + f"_da_{int(experiment_params['data_augmentation'])}" \
                        + f"_ts_{experiment_params['target_size'][0]}" \
                        + f"_{experiment_params['target_size'][1]}" \
                        + f"_t_{int(experiment_params['trainable'])}" \
                        + f"_s_{int(experiment_params['shuffle'])}" \
                        + f"_bs_{experiment_params['batch_size']}" \
                        + f"_cl_{self.__model_config['model']['cnn']['convolutional_layers']['size']}"
                    
                    # Kernel growth strategy
                    model_instance_id = f"{model_instance_id}_kg_{self.__model_config['model']['cnn']['convolutional_layers']['kernel_growth_strategy']}"

                    # Filter growth strategy
                    model_instance_id = f"{model_instance_id}_fg_{self.__model_config['model']['cnn']['convolutional_layers']['filter_growth_strategy']}"

                    # Local pooling.
                    if self.__model_config['model']['cnn']['local_pooling']['enabled'] == True:
                        model_instance_id = f"{model_instance_id}_lp_{self.__model_config['model']['cnn']['local_pooling']['mode']}"
                    
                    # Global pooling.
                    if self.__model_config['model']['cnn']['global_pooling']['enabled'] == True:
                        model_instance_id = f"{model_instance_id}_gp_{self.__model_config['model']['cnn']['global_pooling']['mode']}"

                    # Spatial dropout.
                    if self.__model_config['model']['cnn']['spatial_dropout']['enabled'] == True:
                        model_instance_id = f"{model_instance_id}_sd_{str(self.__model_config['model']['cnn']['spatial_dropout']['rate']).replace('.', '')}"

                    # Fully connected dropout.
                    if self.__model_config['model']['cnn']['fully_connected_dropout']['enabled'] == True:
                        model_instance_id = f"{model_instance_id}_fcd_{str(self.__model_config['model']['cnn']['fully_connected_dropout']['rate']).replace('.', '')}"

                    model_instance_id = f"{model_instance_id}_all_classes_binary"

                else:
                    model_instance_id = f"{experiment_params['model_id']}" \
                        + f"_da_{int(experiment_params['data_augmentation'])}" \
                        + f"_ts_{experiment_params['target_size'][0]}" \
                        + f"_{experiment_params['target_size'][1]}" \
                        + f"_t_{int(experiment_params['trainable'])}" \
                        + f"_s_{int(experiment_params['shuffle'])}" \
                        + f"_bs_{experiment_params['batch_size']}" \
                        + f"_cl_{self.__model_config['model']['cnn']['convolutional_layers']['size']}"
                    
                    # Local pooling.
                    if self.__model_config['model']['cnn']['local_pooling']['enabled'] == True:
                        model_instance_id = f"{model_instance_id}_lp_{self.__model_config['model']['cnn']['local_pooling']['mode']}"
                    
                    # Global pooling.
                    if self.__model_config['model']['cnn']['global_pooling']['enabled'] == True:
                        model_instance_id = f"{model_instance_id}_gp_{self.__model_config['model']['cnn']['global_pooling']['mode']}"

                    # Spatial dropout.
                    if self.__model_config['model']['cnn']['spatial_dropout']['enabled'] == True:
                        model_instance_id = f"{model_instance_id}_sd_{str(self.__model_config['model']['cnn']['spatial_dropout']['rate']).replace('.', '')}"

                    # Fully connected dropout.
                    if self.__model_config['model']['cnn']['fully_connected_dropout']['enabled'] == True:
                        model_instance_id = f"{model_instance_id}_fcd_{str(self.__model_config['model']['cnn']['fully_connected_dropout']['rate']).replace('.', '')}"
                    
                    model_instance_id = f"{model_instance_id}_{'_'.join(experiment_params['filter_classes'])}"
                    model_instance_id = f"{model_instance_id}_binary"
            
            if "DenseNet" in self.__model_id:
                if self.__filter_classes is None:
                    model_instance_id = f"{experiment_params['model_id']}" \
                        + f"_da_{int(experiment_params['data_augmentation'])}" \
                        + f"_ts_{experiment_params['target_size'][0]}" \
                        + f"_{experiment_params['target_size'][1]}" \
                        + f"_t_{int(experiment_params['trainable'])}" \
                        + f"_s_{int(experiment_params['shuffle'])}" \
                        + f"_bs_{experiment_params['batch_size']}" \
                        + f"_all_classes" \
                        + "_binary"
                else:
                    model_instance_id = f"{experiment_params['model_id']}" \
                        + f"_da_{int(experiment_params['data_augmentation'])}" \
                        + f"_ts_{experiment_params['target_size'][0]}" \
                        + f"_{experiment_params['target_size'][1]}" \
                        + f"_t_{int(experiment_params['trainable'])}" \
                        + f"_s_{int(experiment_params['shuffle'])}" \
                        + f"_bs_{experiment_params['batch_size']}" \
                        + f"_{'_'.join(experiment_params['filter_classes'])}" \
                        + "_binary"

            log.info(f"Experiment::run:: executing experiment with id: {model_instance_id}")

            self.__model_config["model"]["model_id"] = experiment_params["model_id"]
            self.__model_config["model"]["model_instance_id"] = model_instance_id
            self.__model_config["model"]["target_size"] = experiment_params["target_size"]
            self.__model_config["model"]["channels_size"] = experiment_params["channels_size"]
            self.__model_config["model"]["classes_size"] = experiment_params["classes_size"]
            self.__model_config["training"]["class_weights"] = class_weights
            
            if "DenseNet" in self.__model_id:
                dense_net = DenseNet.DenseNet(self.__model_config)
                dense_net = dense_net.build()
            
                self.__histories[model_instance_id] = dense_net.fit(train_data_frame_iterator, validation_data_frame_iterator)

            if "CNN" in self.__model_id:
                cnn = CNN.CNN(self.__model_config)
                cnn = cnn.build()
            
                self.__histories[model_instance_id] = cnn.fit(train_data_frame_iterator, validation_data_frame_iterator)

            # Save training history plots.
            loss_val_loss_plot = plot_training_history(
                histories = {self.__model_id: self.__histories[model_instance_id]}, 
                metrics = ["loss", "val_loss"], 
                epochs = 50, 
                title = f"{self.__model_config['model']['model_instance_id']}\n\nloss / val_loss", 
                x_label = "Epoch", 
                y_label = "Loss", 
                log_y = False, 
                figsize = (16, 10)
            )

            loss_val_loss_plot.savefig(
                f"{pathlib.Path(os.path.join(pathlib.Path.cwd().parents[0], os.path.normpath(Env.OUTPUT_REPORT_IMAGES), self.__model_config['model']['model_instance_id'] + '_loss_val_loss.png'))}"
            )

            accuracy_val_accuracy_plot = plot_training_history(
                histories = {self.__model_id: self.__histories[model_instance_id]}, 
                metrics = ["accuracy", "val_accuracy"], 
                epochs = 50, 
                title = f"{self.__model_config['model']['model_instance_id']}\n\naccuracy / val_accuracy", 
                x_label = "Epoch", 
                y_label = "Loss", 
                log_y = False, 
                figsize = (16, 10)
            )

            accuracy_val_accuracy_plot.savefig(
                f"{pathlib.Path(os.path.join(pathlib.Path.cwd().parents[0], os.path.normpath(Env.OUTPUT_REPORT_IMAGES), self.__model_config['model']['model_instance_id'] + '_accuracy_val_accuracy.png'))}"
            )

        return self

    @property
    def experiment_params(self):
        return self.__experiment_params

    @property
    def model_config(self):
        return self.__model_config

    @property
    def histories(self):
        return self.__histories