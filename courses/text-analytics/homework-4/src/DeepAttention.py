'''
    AUEB M.Sc. in Data Science (part-time)
    Course: Text Analytics
    Semester: Spring 2019
    Subject: Homework 1
        - Alexandros Kaplanis (https://github.com/AlexcapFF/)
        - Spiros Politis
        - Manos Proimakis (https://github.com/manosprom)

    Date: 10/06/2019

    Homework 4: Text classification with RNNs.

    Disclaimer: Keras layer code adopted from M. Kyriakakis in-class demo code.
'''

import numpy as np

import keras
import keras.layers.core

def dot_product(x, kernel):
    '''
        Wrapper for dot product operation, in order to be compatible with both
        Theano and Tensorflow
        Args:
            x (): input
            kernel (): weights
        Returns:
    '''
    if keras.backend.backend() == 'tensorflow':
        return keras.backend.squeeze(keras.backend.dot(x, keras.backend.expand_dims(kernel)), axis=-1)
    else:
        return keras.backend.dot(x, kernel)

'''
    Implements a deep attention layer.
'''
class DeepAttention(keras.layers.core.Layer):
    '''
        Constructor
    '''
    def __init__(
        self,
        kernel_regularizer=None, 
        u_regularizer=None, 
        bias_regularizer=None,
        W_constraint=None, 
        u_constraint=None, 
        b_constraint=None,
        bias=True,
        return_attention=False,
        **kwargs
    ):

        self.supports_masking = True
        self.init = keras.initializers.get('glorot_uniform')

        self.W_regularizer = keras.regularizers.get(kernel_regularizer)
        self.u_regularizer = keras.regularizers.get(u_regularizer)
        self.b1_regularizer = keras.regularizers.get(bias_regularizer)
        self.b2_regularizer = keras.regularizers.get(bias_regularizer)

        self.W_constraint = keras.constraints.get(W_constraint)
        self.u_constraint = keras.constraints.get(u_constraint)
        self.b1_constraint = keras.constraints.get(b_constraint)
        self.b2_constraint = keras.constraints.get(b_constraint)

        self.bias = bias
        self.return_attention = return_attention
        super(DeepAttention, self).__init__(**kwargs)

        
    
    '''
    '''
    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(
            (input_shape[-1], input_shape[-1],),
            initializer = self.init,
            name = '{}_W'.format(self.name),
            regularizer = self.W_regularizer,
            constraint = self.W_constraint
        )
        
        if self.bias:
            self.b1 = self.add_weight(
                (input_shape[-1],),
                initializer = 'zero',
                name = '{}_b1'.format(self.name),
                regularizer = self.b1_regularizer,
                constraint = self.b1_constraint
            )
            
            self.b2 = self.add_weight(
                (1,),
                initializer = 'zero',
                name = '{}_b2'.format(self.name),
                regularizer = self.b2_regularizer,
                constraint = self.b2_constraint
            )
            
        else:
            self.b1 = None
            self.b2 = None

        self.u = self.add_weight(
            (input_shape[-1],),
            initializer = self.init,
            name = '{}_u'.format(self.name),
            regularizer = self.u_regularizer,
            constraint = self.u_constraint
        )

        self.built = True



    '''
    '''
    def compute_mask(self, inputs, mask=None):
        # Do not pass the mask to the next layers
        if self.return_attention:
            return [None, None]
        
        return None

    

    '''
    '''
    def call(self, x, mask = None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b1

        uit = keras.backend.tanh(uit)
        
        eij = dot_product(uit, self.u)
        if self.bias:
            eij += self.b2

        # Apply mask
        if mask is not None:
            eij *= keras.backend.cast(mask, keras.backend.floatx())

        a = keras.backend.expand_dims(keras.backend.softmax(eij, axis = -1))
        
        weighted_input = x * a
        result = keras.backend.sum(weighted_input, axis = 1)

        if self.return_attention:
            return [result, a]
        
        return result

    
    
    '''
    '''
    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]