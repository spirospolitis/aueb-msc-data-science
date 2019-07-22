#!/bin/bash

# Remove output dir before run
hdfs dfs -rm -r /data/part-3/

# Execute map-reduce job
hadoop jar ../hadoop-streaming-2.6.0.jar -file mapper.py -mapper mapper.py -file reducer.py -reducer reducer.py -input /data/shakespeare.txt -output /data/part-3