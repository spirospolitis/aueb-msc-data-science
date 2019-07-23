'''
    M.Sc. in Data Science
    Course: Text Analytics
    Professor: I. Androutsopoulos
    Subject: Homework 1
    Team: Alexandros Kaplanis (p3351802), Spiros Politis (p3351814), Manos Proimakis (p3351815)

    File name: Vocabulary.py
    Date created: 06/05/2019
'''

class Vocabulary(object):
    def __init__(self, cutoff_thresshold=10, cutoff_replacement="*UNK*"):
        self._cutoff_thresshold = cutoff_thresshold
        self._cutoff_replacement = cutoff_replacement
            
    @property
    def counts(self):
        return self._counts
    
    @property
    def cutoff_counts(self):
        from collections import Counter
        dic = { x: self._counts[x] for x in self._counts if self._counts[x] >= self._cutoff_thresshold }
        return Counter(dic)
    
    @property
    def size(self):
        return self._size
    
    def clean_sentence(self, tokenized_sentence:list):
        return [word if word in self._vocabulary else self._cutoff_replacement for word in tokenized_sentence]
    
    @property
    def unique(self):
        return self._unique
    
    def __generate_word_counts_from_corpus(self, tokenized_sentences: list):        
        from collections import Counter
        word_counter = Counter()        
        for sentence in tokenized_sentences:
            word_counter.update(sentence)
        return word_counter
    
    def fit(self, corpus:str = None, counts = None):       
        from nltk import sent_tokenize
        sentences = sent_tokenize(corpus)
        
        from nltk import TweetTokenizer
        tweet_wt = TweetTokenizer()
        sentences = [tweet_wt.tokenize(sent) for sent in sentences]
        
        from nltk.lm import Vocabulary
        if(sentences is not None):
            counts = self.__generate_word_counts_from_corpus(sentences)
        
        if (counts is None):
            raise Exception("Invalid arguments exception")
        
        self._counts = counts
        self._vocabulary = Vocabulary(
                                       counts = self.counts,
                                       unk_cutoff = self._cutoff_thresshold,
                                       unk_label = self._cutoff_replacement
                                      )
        
        self._unique = list(self._vocabulary)
        self._size = len(self._unique)