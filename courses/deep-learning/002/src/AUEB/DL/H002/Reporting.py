import logging
import numpy as np
import os
import pandas as pd
import pathlib
import sklearn
import tensorflow as tf
from tqdm import tqdm

from . import Env
from . import Logger
from . import Visualization
from .Data import Ingest

log = logging.getLogger("msc-ds-dl-h-002")

def training_logs_to_df(
    architecture: str
):
    logs_path = pathlib.Path(os.path.join(pathlib.Path.cwd().parents[0], os.path.normpath(Env.OUTPUT_MODEL_BASE_DIR), os.path.normpath(Env.OUTPUT_MODEL_LOGS_DIR)))

    log_dfs = []

    for log_path in logs_path.rglob(architecture + "*.*"):

        model_id =  os.path.split(log_path)[1].replace("_binary_training.csv", "")
        parse_log_filename = os.path.split(log_path)[1].split("_")

        if architecture == "CNN":
            if "all" not in parse_log_filename:
                architecture_param = parse_log_filename[0]
                data_augmentation_param = bool(int(parse_log_filename[2]))
                layers_param = parse_log_filename[13]
                local_pooling_param = parse_log_filename[15]
                global_pooling_param = parse_log_filename[17]
                target_class_param = parse_log_filename[22] + "_" + parse_log_filename[23]

                try:
                    log_df = pd.read_csv(log_path, sep = "\t")
                    log_df["architecture"] = architecture_param
                    log_df["data_augmentation"] = data_augmentation_param
                    log_df["layers"] = layers_param
                    log_df["local_pooling"] = local_pooling_param
                    log_df["global_pooling"] = global_pooling_param
                    log_df["target_class"] = target_class_param
                    log_df["model_id"] = model_id
                    
                    log_dfs.append(log_df)
                except Exception as error:
                    print(error)
            else:
                architecture_param = parse_log_filename[0]
                data_augmentation_param = bool(int(parse_log_filename[2]))
                layers_param = parse_log_filename[13]
                local_pooling_param = parse_log_filename[20]
                global_pooling_param = parse_log_filename[22]
                target_class_param = parse_log_filename[27] + "_" + parse_log_filename[28]

                try:
                    log_df = pd.read_csv(log_path, sep = "\t")
                    log_df["architecture"] = architecture_param
                    log_df["data_augmentation"] = data_augmentation_param
                    log_df["layers"] = layers_param
                    log_df["local_pooling"] = local_pooling_param
                    log_df["global_pooling"] = global_pooling_param
                    log_df["target_class"] = target_class_param
                    log_df["model_id"] = model_id
                    
                    log_dfs.append(log_df)
                except Exception as error:
                    print(error)

        elif architecture == "DenseNet":
            if "all" not in parse_log_filename:
                architecture_param = parse_log_filename[0]
                data_augmentation_param = bool(int(parse_log_filename[2]))
                target_class_param = parse_log_filename[12] + "_" + parse_log_filename[13]

                try:
                    log_df = pd.read_csv(log_path, sep = "\t")
                    log_df["architecture"] = architecture_param
                    log_df["data_augmentation"] = data_augmentation_param
                    log_df["target_class"] = target_class_param
                    log_df["model_id"] = model_id

                    log_dfs.append(log_df)
                except Exception as error:
                    print(error)
            else:
                architecture_param = parse_log_filename[0]
                data_augmentation_param = bool(int(parse_log_filename[2]))
                target_class_param = parse_log_filename[12] + "_" + parse_log_filename[13]

                try:
                    log_df = pd.read_csv(log_path, sep = "\t")
                    log_df["architecture"] = architecture_param
                    log_df["data_augmentation"] = data_augmentation_param
                    log_df["target_class"] = target_class_param
                    log_df["model_id"] = model_id

                    log_dfs.append(log_df)
                except Exception as error:
                    print(error)

    return pd.concat(log_dfs)

def evaluate_to_df(
    architecture: str = None
):
    # Get the test data generator.
    test_data_frame_iterator = Ingest.test_binary(
        batch_size = 32, 
        target_size = (224, 224), 
        shuffle = False, 
        filter_classes = None
    )

    model_checkpoints_path = pathlib.Path(os.path.join(pathlib.Path.cwd().parents[0], os.path.normpath(Env.OUTPUT_MODEL_BASE_DIR), os.path.normpath(Env.OUTPUT_MODEL_CHECKPOINT_DIR))) 
    report_summaries_path = pathlib.Path(os.path.join(pathlib.Path.cwd().parents[0], os.path.normpath(Env.OUTPUT_REPORT_SUMMARIES)))
    
    dfs = []
    
    if architecture == None:
        model_checkpoints_path = model_checkpoints_path.rglob("*.*")
    else:
        model_checkpoints_path = model_checkpoints_path.rglob(architecture + "*.*")
    
    for model_checkpoint_path in model_checkpoints_path:
        model_id = os.path.split(model_checkpoint_path)[1].replace(".hdf5", "")
 
        log.info(f"Evaluating {model_id}")
    
        model = tf.keras.models.load_model(
            model_checkpoint_path
        )

        test_loss, test_accuracy, test_auc = model.evaluate(test_data_frame_iterator)

        df = pd.DataFrame(
            np.array(
                [[model_id, test_loss, test_accuracy, test_auc]]
            ),
            columns = [
                "model_id", 
                "test_loss", 
                "test_accuracy", 
                "test_auc"
            ]
        )
        
        display(df)

        dfs.append(df)

        del model

    return pd.concat(dfs)

def prediction_report_to_df(architecture: str = None):
    # Get the test data generator.
    test_data_frame_iterator = Ingest.test_binary(
        batch_size = 32, 
        target_size = (224, 224), 
        shuffle = False, 
        filter_classes = None
    )
    
    model_checkpoints_path = pathlib.Path(os.path.join(pathlib.Path.cwd().parents[0], os.path.normpath(Env.OUTPUT_MODEL_BASE_DIR), os.path.normpath(Env.OUTPUT_MODEL_CHECKPOINT_DIR))) 
    report_summaries_path = pathlib.Path(os.path.join(pathlib.Path.cwd().parents[0], os.path.normpath(Env.OUTPUT_REPORT_SUMMARIES)))
    
    dfs = []
    
    if architecture == None:
        model_checkpoints_path = model_checkpoints_path.rglob("*.*")
    else:
        model_checkpoints_path = model_checkpoints_path.rglob(architecture + "*.*")
    
    for model_checkpoint_path in model_checkpoints_path:
        model_id = os.path.split(model_checkpoint_path)[1].replace(".hdf5", "")
 
        log.info(f"Evaluating {model_id}")
    
        model = tf.keras.models.load_model(
            model_checkpoint_path
        )
        
        # Compute true and predicted classes.
        y_true = test_data_frame_iterator.classes
        y_pred = model.predict(test_data_frame_iterator)
        y_pred = tf.math.argmax(
            input = y_pred, 
            axis = 1, 
            output_type = tf.dtypes.int64, 
            name = "y_pred"
        ).numpy()
        
        # Compute classification report.
        classification_report = sklearn.metrics.classification_report(
            y_true, 
            y_pred, 
            labels = [0, 1], 
            target_names = ["negative", "positive"], 
            output_dict = True
        )
        
        accuracy = classification_report["accuracy"]
        precision = classification_report["macro avg"]["precision"]
        recall = classification_report["macro avg"]["recall"]
        f1_score = classification_report["macro avg"]["f1-score"]

        # Compute Cohen Kappa.
        cohen_kappa = sklearn.metrics.cohen_kappa_score(
            y_true, 
            y_pred, 
            labels = [0, 1]
        )
        
        df = pd.DataFrame(
            data = [
                [model_id, accuracy, precision, recall, f1_score, cohen_kappa]
            ], 
            columns = [
                "model_id", "accuracy", "precision", "recall", "f1_score", "cohen_kappa"
            ]
        )
        
        display(df)
        
        dfs.append(df)

        del model

    return pd.concat(dfs)


def generate_confusion_matrices(architecture: str = None):
    # Get the test data generator.
    test_data_frame_iterator = Ingest.test_binary(
        batch_size = 32, 
        target_size = (224, 224), 
        shuffle = False, 
        filter_classes = None
    )
    
    model_checkpoints_path = pathlib.Path(os.path.join(pathlib.Path.cwd().parents[0], os.path.normpath(Env.OUTPUT_MODEL_BASE_DIR), os.path.normpath(Env.OUTPUT_MODEL_CHECKPOINT_DIR))) 
    report_images_path = pathlib.Path(os.path.join(pathlib.Path.cwd().parents[0], os.path.normpath(Env.OUTPUT_REPORT_IMAGES)))
    
    if architecture == None:
        model_checkpoints_path = model_checkpoints_path.rglob("*.*")
    else:
        model_checkpoints_path = model_checkpoints_path.rglob(architecture + "*.*")
    
    for model_checkpoint_path in model_checkpoints_path:
        model_id = os.path.split(model_checkpoint_path)[1].replace(".hdf5", "")
 
        log.info(f"Producing confusion matrix for {model_id}")
    
        model = tf.keras.models.load_model(
            model_checkpoint_path
        )
        
        # Compute true and predicted classes.
        y_true = test_data_frame_iterator.classes
        y_pred = model.predict(test_data_frame_iterator)
        y_pred = tf.math.argmax(
            input = y_pred, 
            axis = 1, 
            output_type = tf.dtypes.int64, 
            name = "y_pred"
        ).numpy()
        
        # Confusion matrix.
        confusion_matrix = Visualization.plot_confusion_matrix(
            y_true, 
            y_pred, 
            labels = ["negative", "positive"], 
            title = f"Confusion matrix\n\n{model_id}"
        )

        confusion_matrix.savefig(
            f"{pathlib.Path(os.path.join(pathlib.Path.cwd().parents[0], os.path.normpath(Env.OUTPUT_REPORT_IMAGES), model_id + '_confusion_matrix.png'))}"
        )

        del model

    return None

# def classification_report_to_df(y_true, y_pred):
#     # Get the test data generator.
#     test_data_frame_iterator = Ingest.test_binary(
#         batch_size = 32, 
#         target_size = (224, 224), 
#         shuffle = False, 
#         filter_classes = None
#     )

#     model_checkpoints_path = pathlib.Path(os.path.join(pathlib.Path.cwd().parents[0], os.path.normpath(Env.OUTPUT_MODEL_BASE_DIR), os.path.normpath(Env.OUTPUT_MODEL_CHECKPOINT_DIR))) 
#     report_summaries_path = pathlib.Path(os.path.join(pathlib.Path.cwd().parents[0], os.path.normpath(Env.OUTPUT_REPORT_SUMMARIES)))

#     dfs = []

    
    
#     classification_report = sklearn.metrics.classification_report(
#         y_true, 
#         y_pred, 
#         labels = [0, 1], 
#         target_names = ["negative", "positive"], 
#         output_dict = True
#     )
    
#     df = pd.DataFrame(classification_report).transpose()
#     df = classification_report_df.apply(lambda x: np.round(x, 3))
#     df.loc["accuracy", "precision"] = ""
#     df.loc["accuracy", "recall"] = ""
#     df.loc["accuracy", "support"] = ""
    
#     return classification_report_df