#!/usr/bin/env python
import sys
import string
import itertools

for i, line in enumerate(sys.stdin):
    # Strip the line of text from leading and trailing white space
    line = line.strip()

    # Remove punctuation
    line = line.translate(None, string.punctuation)

    # Convert all words to lowercase
    line = line.lower()

    # Split line into words
    words = line.split()

    # Create combinations of words found in the line
    word_combinations = itertools.combinations(words, 2)

    for word_combination in word_combinations:
        # Produce a key with format
        # <tupple 1st word>-<tupple 2nd word>
        # Tab (\t) demarkates key, value
        print("({},{})\t{}".format(word_combination[0], word_combination[1], "1"))