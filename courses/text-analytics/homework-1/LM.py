'''
    M.Sc. in Data Science
    Course: Text Analytics
    Professor: I. Androutsopoulos
    Subject: Homework 1
    Team: Alexandros Kaplanis (p3351802), Spiros Politis (p3351814), Manos Proimakis (p3351815)

    File name: LM.py
    Date created: 06/05/2019
'''

from sklearn.base import BaseEstimator, ClassifierMixin

from Preprocessor import Preprocessor
from SentencePadding import SentencePadding
from Vocabulary import Vocabulary

class LM(BaseEstimator):    
    def __init__(self, vocabulary: Vocabulary, sentence_padding : SentencePadding = None, preprocessor: Preprocessor= None, alpha=1, rank = 2):
        self.__alpha = alpha
        
        if rank < 1:
            raise ValueError("rank should be higher than 1")
        
        self._rank = rank
        self._sentence_padding = sentence_padding
        self._vocabulary = vocabulary
        
        self._init_counters()

    @property
    def rank(self):
        return self._rank
    
    @property
    def counters(self):
        return self._counters
    
    def fit(self, train_corpus, verbose=False):
        self._init_counters()
        sentences = self._create_sentences(train_corpus)
        for sentence in sentences:
            sentence = self._preprocess(sentence)
            sentence = self._vocabulary.clean_sentence(sentence)
            sentence = self._add_padding(sentence, self._rank - 1)
            self._update_counter(self._rank, sentence)
            self._update_counter(self._rank - 1, sentence)
        
        return self
    
    def predict(self, sentence, verbose=False):
        sentence_prob, idx_count = self._calculate_sentence_prob(sentence, verbose)
        return sentence_prob
    
    def score(self, test_corpus, verbose=False):
        import math
        sentences = self._create_sentences(test_corpus)
        total_prob = 0
        total_count =  0
        for sentence in sentences:
            sentence_prob, sentence_count = self._calculate_sentence_prob(sentence, verbose)
            total_prob += sentence_prob
            total_count += sentence_count
        entropy = -total_prob / total_count
        perplexity = math.pow(2,entropy)
        return entropy, perplexity
    
    def _calculate_sentence_prob(self, sentence, verbose=False):
        sentence = self._preprocess(sentence)
        sentence = self._vocabulary.clean_sentence(sentence)
        sentence = self._add_padding(sentence, self._rank)
        
        import math
        sum_prob = 0
        idx_count = 0;
        for idx in range(self._rank - 1,len(sentence)):
            prob = self._calculate_idx_prob(sentence, idx)
            log_prob = math.log2(prob)
            self._print({"logprob": log_prob})
            sum_prob += log_prob
            idx_count+=1
        return sum_prob, idx_count
    
    def _calculate_idx_prob(self, sentence, idx, verbose = False):
        self._print("=======================================================================", verbose=verbose)
        current_ngram_key = self._create_key(sentence, idx, 0)
        previous_ngram_key = self._create_key(sentence, idx, 1)
        current_ngram_count = self._counters.get(self._rank)[current_ngram_key]
        previous_ngram_count = self._counters.get(self._rank - 1)[previous_ngram_key]

        self._print({"n": (current_ngram_key, current_ngram_count), "n-1" : ( previous_ngram_key, previous_ngram_count) }, verbose=verbose)

        prob = self._laplace_smoothing(current_ngram_count, previous_ngram_count, self.__alpha, self._vocabulary.size)
        self._print({"prob": prob}, verbose=verbose)
        self._print("=======================================================================", verbose=verbose)
        return prob
   
    def _laplace_smoothing(self, current_ngram_count, previous_ngram_count, alpha, vocabulary_size):
        numerator = current_ngram_count + self.__alpha
        denominator = previous_ngram_count + (alpha * vocabulary_size)
        self._print({ "numerator": numerator, "denominator": denominator, "alpha": self.__alpha, "vocabulary_size": vocabulary_size })
        return numerator / denominator

    def _create_key(self, sentence, index, to):
        key = ()
        for i in range (self._rank - 1, to - 1, -1):
            key = (*key, sentence[index - i])
        return key
    
    def _init_counters(self):
        from collections import Counter
        self._counters = { key: Counter() for key in range(self._rank-1, self._rank + 1) }
    
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

    def _add_padding(self, tokenized_sentence, rank = 1):
        return self._sentence_padding.add_padding(tokenized_sentence, times_start = rank, times_end = 1, indexed_start=True, indexed_end=False)
    
    def _update_counter(self, rank, sentence):
        from nltk import ngrams
        counts = [gram for gram in ngrams(sentence, rank)]
        self._counters.get(rank).update(counts)
        
    def _print(self, *args, **kargs):
        if kargs.get("verbose", False):
            print(args)