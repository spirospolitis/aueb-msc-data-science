#!/usr/bin/env python
import sys

# Holds the tuples counts
tuple_count = {}

# For each line coming in from stdin buffer
for line in sys.stdin:
    # Strip the line of text from leading and trailing white space
    line = line.strip()

    # Retrieve key and related value
    t, count = line.split("\t", 1)
    
    # Count must be parsable to a number
    try:
        count = int(count)
    except ValueError:
        continue

    # For each key encountered, increment by the count
    try:
        tuple_count[t] = tuple_count[t] + count
    except:
        tuple_count[t] = count

# Flush to stdout
for t in tuple_count.keys():
    print("{}\t{}".format(t, tuple_count[t]))