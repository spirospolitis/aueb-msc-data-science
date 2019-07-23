'''
    M.Sc. in Data Science
    Course: Text Analytics
    Professor: I. Androutsopoulos
    Subject: Homework 1
    Team: Alexandros Kaplanis (p3351802), Spiros Politis (p3351814), Manos Proimakis (p3351815)

    File name: Preprocessor.py
    Date created: 06/05/2019
'''

class Preprocessor(object):
    def preprocess(self, corpus:str):
        sentences = self._create_sentences(corpus)
        sentences = [self._preprocess(sentence) for sentence in sentences]
        return sentences

    def _create_sentences(self, corpus):
        from nltk import sent_tokenize
        sentences = sent_tokenize(corpus)
        return sentences
    
    def _preprocess(self, sentence):
        sentence = self._normalize(sentence)
        sentence = self._tokenize(sentence)
        return sentence
    
    def _normalize(self, sentence):
        sentence = sentence.lower()
        sentence = sentence.strip()
        return sentence
    
    def _tokenize(self, sentence):
        from nltk.tokenize import TweetTokenizer
        tweet_wt = TweetTokenizer()
        sentence = tweet_wt.tokenize(sentence)
        return sentence