import numpy as np
import tensorflow as tf

class MLP(object):

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
        model_name:str, 
        input_shape:(), 
        hidden_layers_spec:[], 
        output_layer_spec:{}, 
        dropout_layer_spec:{} = None
    ):
        # Input layer (None, 28, 28)
        input_layer = tf.keras.layers.Input(
            shape = input_shape, 
            name = "input_layer"
        )
        
        l = input_layer

        # Flatten layer: iff shape of input is a 3-dimensional tensor.
        if len(input_shape) > 1:
            l = tf.keras.layers.Flatten( 
                name = "flatten_layer_1"
            )(l)

        # Add a dropout layer?
        if dropout_layer_spec != None:
            l = tf.keras.layers.Dropout( 
                rate = dropout_layer_spec["rate"], 
                name = f"dropout_layer_1"
            )(l)

        # Fully-connected hidden layer(s).
        for i, hidden_layer_spec in enumerate(hidden_layers_spec):
            l = tf.keras.layers.Dense(
                units = hidden_layer_spec["units"],
                kernel_initializer = hidden_layer_spec["kernel_initializer"],
                activation = hidden_layer_spec["activation"],
                name = f"hidden_layer_{i + 1}"
            )(l)

            # Add a dropout layer?
            if dropout_layer_spec != None:
                l = tf.keras.layers.Dropout( 
                    rate = dropout_layer_spec["rate"], 
                    name = f"dropout_layer_{i + 2}"
                )(l)
        
        output_layer = tf.keras.layers.Dense(
            units = output_layer_spec["units"], 
            kernel_initializer = output_layer_spec["kernel_initializer"],
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