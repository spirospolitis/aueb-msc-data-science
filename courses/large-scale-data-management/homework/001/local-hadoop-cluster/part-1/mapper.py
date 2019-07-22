#!/usr/bin/env python
import sys
import string

for i, line in enumerate(sys.stdin):
    # Strip the line of text from leading and trailing white space
    line = line.strip()

    # Remove punctuation
    line = line.translate(None, string.punctuation)

    # Convert all words to lowercase
    line = line.lower()

    # Split line into words
    words = line.split()

    # Counter demarkating tuple bounds (x, x + 1), x = current word
    i = 0

    while(i < len(words) - 1):
        # Produce a key with format
        # <word at pos x>-<word at pos x + 1>
        # Tab (\t) demarkates key, value
        print("({},{})\t{}".format(words[i], words[i + 1], "1"))
        
        i += 1