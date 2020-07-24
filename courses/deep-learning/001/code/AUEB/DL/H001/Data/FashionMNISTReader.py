"""
    AUEB M.Sc. in Data Science
    Semester: Sprint 2020
    Course: Deep Learning
    Homework: 1
    Lecturer: P. Malakasiotis
    Author: Spiros Politis
"""
import os
import gzip
import logging
import numpy as np

log = logging.getLogger("msc-ds-dl-h-001")

"""
    Load MNIST data from 'path'.

    Note: Direct copy from https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
"""
def load_mnist(path:str, kind:str = "train"):
    
    labels_path = os.path.join(
        path,
        "%s-labels-idx1-ubyte.gz" % kind
    )

    images_path = os.path.join(
        path,
        "%s-images-idx3-ubyte.gz" % kind
    )

    try:
        with gzip.open(labels_path, "rb") as lbpath:
            labels = np.frombuffer(
                lbpath.read(), 
                dtype = np.uint8,
                offset = 8
            )

        with gzip.open(images_path, "rb") as imgpath:
            images = np.frombuffer(
                imgpath.read(), 
                dtype = np.uint8,
                offset = 16
            ).reshape(len(labels), 784)

        log.info(f"Done loading Fashion-MNIST {kind} data")
    except Exception as error:
        log.error(f"{error}")

    return images, labels