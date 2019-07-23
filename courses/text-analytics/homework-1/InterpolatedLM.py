'''
    M.Sc. in Data Science
    Course: Text Analytics
    Professor: I. Androutsopoulos
    Subject: Homework 1
    Team: Alexandros Kaplanis (p3351802), Spiros Politis (p3351814), Manos Proimakis (p3351815)

    File name: InterpolatedLM.py
    Date created: 06/05/2019
'''

from LM import LM

class InterpolatedLM(object):
    def __init__(self, model1 : LM, model2: LM, rank = 2, lamdas = None):
        self.__model1 = model1
        self.__model2 = model2
        self.__lamdas = lamdas
        
    def fit(self, train_corpus):
        self.__model1.fit(train_corpus)
        self.__model2.fit(train_corpus)
        return self
    
    def predict(self, sentence, verbose=False):
        prob_count_model_1 = self.__model1.predict(sentence, verbose)
        prob_count_model_2 = self.__model2.predict(sentence, verbose)
        prob = (self.__lamdas[0] * prob_count_model_2 + (1-self.__lamdas[1]) * prob_count_model_1)
        return prob
    
    def score(self, test_corpus, verbose=False):
        import math
        sentences = self._create_sentences(test_corpus)

        total_prob = 0
        total_count =  0
        for sentence in sentences:
            prob_count_model_1, idx_count_model_1 = self.__model1._calculate_sentence_prob(sentence, verbose)
            prob_count_model_2, idx_count_model_2 = self.__model2._calculate_sentence_prob(sentence, verbose)
            prob = (self.__lamdas[0] * prob_count_model_2 + (1-self.__lamdas[1]) * prob_count_model_1)
            total_prob += prob
            total_count += idx_count_model_2
        entropy = -total_prob / total_count
        perplexity = math.pow(2,entropy)
        return entropy, perplexity
        
    def _create_sentences(self, corpus):
        from nltk import sent_tokenize
        sentences = sent_tokenize(corpus)
        return sentences