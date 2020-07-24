"""
    AUEB M.Sc. in Data Science
    Semester: Sprint 2020
    Course: Deep Learning
    Homework: 1
    Lecturer: P. Malakasiotis
    Author: Spiros Politis
"""

"""
    Illustrates the vanishing / exploding gradients problem.
"""
def gradients(minval = 0, maxval = 1, passes = 1):
    import numpy as np
    import tensorflow as tf

    # Initialize a random uniform [0, 1] tensor of weights.
    weights = tf.random.uniform(
        (512, 512), minval = minval, maxval = maxval, dtype = tf.dtypes.float32, seed = 19730618, name = "weights_tensor"
    )
    
    # Initialize a random uniform [0, 1] tensor of gradients.
    gradients = tf.random.uniform(
        (512, 512), minval = minval, maxval = maxval, dtype = tf.dtypes.float32, seed = 19730618, name = "gradients_tensor"
    )
    
    # Simulate 100 passes.
    for i in range(passes):    
        weights = tf.math.multiply(
            weights, gradients, name = "weights_multiply_tensor"
        )

    # Compute the tensor mean.
    mean = tf.math.reduce_mean(
        weights, axis = None, keepdims = False, name = "weights_mean"
    ).numpy(),

    # Compute the tensor std.
    std = tf.math.reduce_std(
        weights, axis = None, keepdims = False, name = "weights_std"
    ).numpy().std()

    return mean, std