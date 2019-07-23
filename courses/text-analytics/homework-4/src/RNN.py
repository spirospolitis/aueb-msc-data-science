'''
    AUEB M.Sc. in Data Science (part-time)
    Course: Text Analytics
    Semester: Spring 2019
    Subject: Homework 1
        - Alexandros Kaplanis (https://github.com/AlexcapFF/)
        - Spiros Politis
        - Manos Proimakis (https://github.com/manosprom)

    Date: 10/06/2019

    Homework 4: Text classification with RNNs
'''

import os

import warnings
import sklearn.exceptions

import keras

warnings.filterwarnings("ignore", category = sklearn.exceptions.UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category = FutureWarning)

# Remove deprecation warnings because of TF 1.14.0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class RNN:
    '''
        Constructor
    '''
    def __init__(self):
        self.__inputs = None
        self.__outputs = None
        self.__model = None
        self.__history = None
        
        

    '''
    '''
    def input_layer(
        self, 
        input_size:int = 0, 
        name:str = None
    ):
        import keras.layers
        import keras.models

        self.__inputs = keras.layers.Input(
            shape = (input_size, ), 
            name = name
        )
        
        return self
    


    '''
    '''
    def embeddings_layer(
        self, 
        input_dim:int = 0, 
        output_dim:int = 0, 
        weights = None, 
        input_length = 0, 
        name:str = None
    ):
        import keras.layers.embeddings

        self.__outputs = keras.layers.embeddings.Embedding(
            input_dim, 
            output_dim, 
            weights = [weights], 
            input_length = input_length,
            mask_zero = True, 
            trainable = False, 
            name = name
        )(self.__inputs)

        return self



    '''
    '''
    def dropout_layer(
        self, 
        rate:float = 0.2, 
        name:str = None, 
        parameters = None, 
        parameter_names:[] = None
    ):
        import keras.layers
        
        if parameters is None:
            self.__outputs = keras.layers.Dropout(
                rate = rate, 
                name = name
            )(self.__outputs)
        else:
            self.__outputs = keras.layers.Dropout(
                rate = parameters[parameter_names[0]], 
                name = name
            )(self.__outputs)

        return self



    '''
    '''
    def dense_layer(
        self, 
        units:int = 0, 
        activation:str = 'relu', 
        name:str = None, 
        parameters = None, 
        parameter_names:[] = None
    ):
        import keras.layers
        
        if parameters is None:
            self.__outputs = keras.layers.Dense(
                units, 
                activation = activation, 
                name = name
            )(self.__outputs)
        else:
            self.__outputs = keras.layers.Dense(
                parameters[parameter_names[0]], 
                activation = parameters[parameter_names[1]], 
                name = name
            )(self.__outputs)
        
        return self



    '''
    '''
    def gru_layer(
        self, 
        units:int = 0, 
        activation:str = 'tanh', 
        return_sequences:bool = False, 
        recurrent_dropout:float = 0.2, 
        name:str = None,
        parameters = None, 
        parameter_names:[] = None
    ):
        import keras.layers 
        import keras.layers.recurrent

        if parameters is None:    
            self.__outputs = keras.layers .Bidirectional(
                keras.layers.recurrent.GRU(
                    units, 
                    return_sequences = return_sequences, 
                    recurrent_dropout = recurrent_dropout
                ),
                name = name
            )(self.__outputs)
        else:
            self.__outputs = keras.layers .Bidirectional(
            keras.layers.recurrent.GRU(
                parameters[parameter_names[0]], 
                return_sequences = return_sequences, 
                recurrent_dropout = parameters[parameter_names[1]]
            ),
            name = name
        )(self.__outputs)
        
        return self



    '''
    '''
    def lstm_layer(
        self, 
        units:int = 0, 
        activation:str = 'tanh', 
        return_sequences:bool = False, 
        recurrent_dropout:float = 0.2, 
        name:str = None
    ):
        import keras.layers 
        import keras.layers.recurrent
        
        self.__outputs = keras.layers.Bidirectional(
            keras.layers.recurrent.LSTM(
                units, 
                return_sequences = return_sequences, 
                recurrent_dropout = recurrent_dropout
            ),
            name = name
        )(self.__outputs)

        return self
    


    '''
    '''
    def linear_attention_layer(
        self, 
        kernel_regularizer = None, 
        bias_regularizer = None,
        W_constraint = None, 
        b_constraint = None,
        bias = True,
        return_attention = False
    ):
        from LinearAttention import LinearAttention

        self.__outputs = LinearAttention(
            kernel_regularizer = kernel_regularizer, 
            bias_regularizer = bias_regularizer,
            W_constraint = W_constraint, 
            b_constraint = b_constraint,
            bias = bias,
            return_attention = return_attention
        )(self.__outputs)

        return self
    


    '''
    '''
    def deep_attention_layer(
        self,
        kernel_regularizer = None, 
        u_regularizer = None, 
        bias_regularizer = None,
        W_constraint = None, 
        u_constraint = None, 
        b_constraint = None,
        bias = True,
        return_attention = False,
        **kwargs
    ):
        from DeepAttention import DeepAttention
        
        self.__outputs = DeepAttention(
            kernel_regularizer = kernel_regularizer, 
            u_regularizer = u_regularizer, 
            bias_regularizer = bias_regularizer,
            W_constraint = W_constraint, 
            u_constraint = u_constraint, 
            b_constraint = b_constraint,
            bias = bias,
            return_attention = return_attention,
            **kwargs
        )(self.__outputs)

        return self



    '''
    '''
    def load(
        self,
        name:str
    ):
        from numpy import loadtxt
        from keras.models import load_model
        
        # Load model
        self.__model = load_model('./models/' + name)

        return self

        

    '''
        Compiles the Keras model.


    '''
    def compile(
        self, 
        loss:str = 'binary_crossentropy', 
        optimizer = None, 
        metrics:[] = None
    ):
        import keras.models

        # Create a Keras model.
        self.__model = keras.models.Model(
            inputs = self.__inputs, 
            outputs = self.__outputs
        )
        
        # Compile the model.
        self.__model.compile(
            loss = 'binary_crossentropy',
            optimizer = optimizer,
            metrics = metrics
        )

        return self



    '''
        Fits the model.

        :param X_train: X training set.
        :param y_train: y training set.
        :param X_validate: X validation set.
        :param y_validate: y validation set.
        :param batch_size: training bacth size.
        :param epochs: number of epochs to train.
        :param callback: list opf Keras callbacks.
        :param shuffle: boolean flag, whether to shuffle the training data or not.
        :param verbose: verbosity level.

        :returns: self.
    '''
    def fit(
        self, 
        X_train, 
        y_train, 
        X_validate, 
        y_validate, 
        batch_size:int = 1000, 
        epochs:int = 50, 
        callbacks:[] = None, 
        shuffle:bool = False, 
        verbose:int = 0
    ):
        self.__history = self.__model.fit(
            X_train, 
            y_train,
            batch_size = batch_size,
            epochs = epochs,
            verbose = verbose,
            callbacks = callbacks,
            validation_data = (X_validate, y_validate),
            shuffle = shuffle
        )

        return self


    
    '''
        Class properties
    '''
    @property
    def inputs(self):
        return self.__inputs

    @property
    def outputs(self):
        return self.__outputs

    @property
    def model(self):
        return self.__model
    
    @property
    def history(self):
        return self.__history