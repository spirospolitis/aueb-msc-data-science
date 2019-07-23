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

import pandas as pd
from keras import backend as K
import matplotlib.pyplot as plt

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

'''
    Ingests a dataset.
    
    :param file_path: Path from which to load a dataset.
    :param num_lines: Maximum number of lines to ingest.
    
    :returns: Pandas dataframe.
'''
def create_dataset(file_path:str, num_lines:int):
    df = pd.read_csv(
        file_path, \
        engine='python', \
        names=[
            'polarity',
            'id',
            'date',
            'query',
            'user',
            'text'
        ],
        index_col='id',
        sep=',', 
        header=None,
        nrows=num_lines,
        encoding = 'latin_1'
    )
    return df



'''
    Samples the dataset, taking up to n rows from every target class.

    :param num_rows_from_each_class: number of rows to sample.

    :returns: Pandas DataFrame
'''
def take_n_samples_from_each_category(df, num_rows_from_each_class):
    some_negative = df[df['polarity'] == 0].sample(num_rows_from_each_class)
    some_positive = df[df['polarity'] == 4].sample(num_rows_from_each_class)
    sample_df = pd.concat([some_negative, some_positive])

    return sample_df



'''
'''
def get_mismatched_tweets(x, y_true, y_pred):
    if( not (len(x) == len(y_true) == len(y_pred))):
        raise 'Invalid Sizes'
    return pd.DataFrame.from_dict([{'text': x[i], 'actual': y_true[i], 'predicted': y_pred[i]} for i in range(len(y_true)) if y_true[i]!=y_pred[i]])



'''
'''
def plot_history(history):
    fig = plt.figure(figsize=(20,5))
    ax1 = fig.add_subplot(1, 3, 1)
    # summarize history for accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='upper left')

    ax2 = fig.add_subplot(1, 3, 2)
    # summarize history for loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper left')

    ax3 = fig.add_subplot(1, 3, 3)
    # summarize history for loss
    ax3.plot(history.history['f1'])
    ax3.plot(history.history['val_f1'])
    ax3.set_title('model f1')
    ax3.set_ylabel('f1')
    ax3.set_xlabel('epoch')
    ax3.legend(['train', 'test'], loc='upper left')



'''
'''
def print_evaluation(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose = 1, batch_size = 100)

    print('\nTest binary cross entropy: %.4f' %  (score[0]))
    print('\nTest precision: %.4f' %  (score[1]))
    print('\nTest recall: %.4f' %  (score[2]))
    print('\nTest f1: %.4f' % (score[3]))
    print('\nTest accuracy: %.4f'% (score[4]))



'''
    Recall metric.

    Only computes a batch-wise average of recall.
    
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
'''
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    
    return recall



'''
    Precision metric.

    Only computes a batch-wise average of precision.
    
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    
    Source
    ------
    https://github.com/fchollet/keras/issues/5400#issuecomment-314747992
'''
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision



'''
    Calculate the F1 score.
'''
def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    
    return 2 * ((p * r) / (p + r))



'''
    Calculate accuracy.
'''
def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=1)
