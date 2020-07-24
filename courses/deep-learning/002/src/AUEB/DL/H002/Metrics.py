"""
    AUEB M.Sc. in Data Science
    Semester: Sprint 2020
    Course: Deep Learning
    Homework: 2
    Lecturer: P. Malakasiotis
    Author: Spiros Politis
"""

import numpy as np
import tensorflow as tf

def tp(y_true: np.ndarray, y_pred: np.ndarray):
    return tf.math.count_nonzero(y_pred * y_true)

def tn(y_true: np.ndarray, y_pred: np.ndarray):
    return tf.math.count_nonzero((y_pred - 1) * (y_true - 1))

def fp(y_true: np.ndarray, y_pred: np.ndarray):
    return tf.math.count_nonzero(y_pred * (y_true - 1))

def fn(y_true: np.ndarray, y_pred: np.ndarray):
    return tf.math.count_nonzero((y_pred - 1) * y_true)

def precision(y_true: np.ndarray, y_pred: np.ndarray):
    return tf.divide(tp(y_true, y_pred), tp(y_true, y_pred) + fp(y_true, y_pred))

def recall(y_true: np.ndarray, y_pred: np.ndarray):
    return tf.divide(tp(y_true, y_pred), tp(y_true, y_pred) + fn(y_true, y_pred))

def f1(y_true: np.ndarray, y_pred: np.ndarray):
    return tf.divide(2 * precision(y_true, y_pred) * recall(y_true, y_pred), precision(y_true, y_pred) + recall(y_true, y_pred))

def kappa():
    pass