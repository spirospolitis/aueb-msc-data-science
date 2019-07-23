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

import re

'''
'''
class Preprocess:
    '''
    '''
    @staticmethod
    def remove_tags(sentence):
        return re.sub(r'@[A-Za-z0-9]+', ' ', sentence)



    '''
    '''
    @staticmethod
    def remove_urls(sentence):
        return re.sub('https?://[A-Za-z0-9./]+', ' ', sentence)



    '''
    '''
    @staticmethod
    def remove_underscores(sentence):
        return re.sub(r'_[A-Za-z0-9]+', ' ', sentence)



    '''
    '''    
    @staticmethod
    def remove_special_characters(sentence):
        return re.sub(r'\W', ' ', sentence)



    '''
    '''
    @staticmethod
    def remove_rem_tags(sentence):
        return re.sub(r'^@\s+', ' ', sentence)



    '''
    '''
    @staticmethod
    def remove_rem_underscores(sentence):
        return re.sub(r'^ _\s+', ' ', sentence)
    


    '''
    '''
    @staticmethod
    def remove_multiple_spaces(sentence):
        return re.sub(r' +', ' ', sentence)



    '''
    '''
    @staticmethod
    def remove_trailing_spaces(sentence):
        return sentence.strip()



'''
'''
def preprocess_row(row):
    row = Preprocess.remove_tags(row)
    row = Preprocess.remove_urls(row)
    row = Preprocess.remove_underscores(row)
    row = Preprocess.remove_special_characters(row)
    row = Preprocess.remove_rem_tags(row)
    row = Preprocess.remove_rem_underscores(row)
    row = Preprocess.remove_multiple_spaces(row)
    row = Preprocess.remove_trailing_spaces(row)
    row = row.lower()
    return row