import numpy as np
import tensorflow as tf

class CNN(object):

    """
    """
    def __init__(self, random_seed: int = None):
        if random_seed != None:
            np.random.seed(random_seed)
            tf.random.set_seed = random_seed
    
    """
    """
    @staticmethod
    def create(
        model_name: str, 
        input_shape: (), 
        conv_layers_spec: [], 
        output_layer_spec: {}
    ):
        # Input layer (None, 28, 28)
        input_layer = tf.keras.layers.Input(
            shape = input_shape, 
            name = "input_layer"
        )
        
        l = input_layer

        # Stack convolutional layers.
        for i, conv_layer_spec in enumerate(conv_layers_spec):
            l = tf.keras.layers.Conv2D(
                filters = conv_layer_spec["filters"],
                kernel_size = conv_layer_spec["kernel_size"],
                strides = conv_layer_spec["strides"],
                padding = conv_layer_spec["padding"],
                dilation_rate = conv_layer_spec["dilation_rate"],
                activation = conv_layer_spec["activation"],
                name = f"conv_2d_layer_{i + 1}"
            )(l)

            l = tf.keras.layers.MaxPool2D(
                pool_size = conv_layer_spec["max_pool_2d"]["pool_size"],
                strides = conv_layer_spec["max_pool_2d"]["strides"],
                padding = conv_layer_spec["max_pool_2d"]["padding"],
                name = f"max_pool_2d_layer_{i + 1}"
            )(l)

            # If droput is defined.
            if "dropout" in conv_layer_spec:
                l = tf.keras.layers.Dropout(
                    rate = conv_layer_spec["dropout"]["rate"],
                    name = f"dropout_layer_{i + 1}"
                )(l)

        # Flatten.
        l = tf.keras.layers.Flatten(
            name = "flatten_layer"
        )(l)
        
        # Define the output layer.
        output_layer = tf.keras.layers.Dense(
            units = output_layer_spec["units"],
            activation = output_layer_spec["activation"],
            name = "output_layer"
        )(l)
        
        model = tf.keras.models.Model(
            inputs = input_layer, 
            outputs = output_layer, 
            name = model_name
        )
        
        return model

    def train(self, X, y, ):
        pass