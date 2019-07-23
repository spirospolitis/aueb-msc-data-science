'''
    M.Sc. in Data Science
    Course: Text Analytics
    Professor: I. Androutsopoulos
    Subject: Homework 1
    Team: Alexandros Kaplanis (p3351802), Spiros Politis (p3351814), Manos Proimakis (p3351815)

    File name: SentencePadding.py
    Date created: 06/05/2019
'''

class SentencePadding(object):
    def __init__(self, pad_word_start=None, pad_word_end=None):
        self.pad_word_start = pad_word_start
        self.pad_word_end = pad_word_end
        
    def _wrap_with_asterisk(self, word, times = None):
        return "*" + word + "*"

    def _gen_pad(self, pad_word, times, index= False):
        if pad_word is None: return []
        return [self._wrap_with_asterisk(pad_word + str(i)) if index else self._wrap_with_asterisk(pad_word) for i in range(times)]

    def add_padding(self, tokenized_sentence: list, times_start: int = 1, times_end: int = 1, indexed_start: bool =False, indexed_end: bool=False):
        return self._gen_pad(self.pad_word_start, times_start, indexed_start) + tokenized_sentence + self._gen_pad(self.pad_word_end, times_end, indexed_end)  