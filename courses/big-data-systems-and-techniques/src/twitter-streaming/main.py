# -*- coding: utf-8 -*-
"""
    File: main.py
    Date: 07/2020
    Author: Spiros Politis
    Python: 3.6
"""
import ast
import json
import pyspark
import pyspark.ml.pipeline
from pyspark import SparkContext
from pyspark.sql import SparkSession, Row
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from Preprocessing.Transformers import LowercaseTransformer, RemoveHTMLTransformer, RemoveNonAlphanumericTransformer

KAFKA_BROKERS = "localhost:9092"
KAFKA_TOPIC = "offers"

"""
    Load the saved trained model from HDFS.
"""
def load_saved_model():
    return pyspark.ml.pipeline.PipelineModel.load("output/one_vs_rest_log_reg.model")

"""
    Applies a text preprocessing pipeline, the same used for modelling.
    :param df: DataFrame
    :param sc: SparkContext
"""
def apply_preprocessing_pipeline(df, sc):
    # Defining pipeline steps.
    lowercase_transformer = LowercaseTransformer.LowercaseTransformer(
        column = "descr"
    )

    remove_html_transformer = RemoveHTMLTransformer.RemoveHTMLTransformer(
        column = "descr",
        sc = sc
    )

    remove_non_alphanumeric_transformer = RemoveNonAlphanumericTransformer.RemoveNonAlphanumericTransformer(
        column = "descr"
    )

    # Defining the pipeline.
    preprocessing_pipeline = pyspark.ml.Pipeline(
        stages = [
            lowercase_transformer,
            remove_html_transformer,
            remove_non_alphanumeric_transformer
        ]
    )

    return preprocessing_pipeline.fit(df).transform(df)

"""
    Applies the entire prediction pipeline on the stream.
    :param rdd: Spark RDD
    :param model: Pipelined model
    :param ss: SparkSession
    :param sc: SparkContext
"""
def apply_pipeline(rdd, model, ss, sc):
    try:
        rdd_row = rdd.map(lambda x: Row(descr = x))
        
        # Convert to RDD row to Dataframe.
        df = ss.createDataFrame(rdd_row)
        
        # Apply text preprocessing pipeline.
        df = apply_preprocessing_pipeline(df, sc)
        
        # Apply prediction pipeline.
        y_pred = model.transform(df)
        
        # Keep only textual description and prediction outcome.
        y_pred = y_pred.select("descr", "prediction")
        
        y_pred.printSchema()
        
        save_predictions(y_pred, sc)
        
        y_pred.show(truncate = False)
    except Exception as error:
        print error

def save_predictions(df, sc):
    # Write Parquet.
    df.write.mode("append").parquet("output/streamed_predictions.parquet")
    
    # Write CSV.
    df.coalesce(1).write.mode("append").csv("output/streamed_predictions.txt")

if __name__ == "__main__":
    # Load best saved model.
    model = load_saved_model()

    # Create Spark session.
    ss = SparkSession.builder.appName("BDSAT Task 5").getOrCreate()
    
    # Get Spark context.
    # sc = SparkContext.getOrCreate()
    sc = ss.sparkContext
    
    # Get a Spark Streaming context with a 10 second window.
    ssc = StreamingContext(sc, 10)
    
    # Create Kafka stream.
    kafka_stream = KafkaUtils.createDirectStream(
        ssc,
        [KAFKA_TOPIC],
        {"metadata.broker.list": KAFKA_BROKERS}
    )
    
    # Get DStream.
    dstream = kafka_stream.map(lambda x: ast.literal_eval(x[1]))
    
    # Convert tweet from JSON to Python dict..
    # dstream = dstream.map(lambda x: json.loads(x, encoding = "utf-8"))
    
    # Extract tweet text from JSON.
    dstream = dstream.map(lambda x: x["tweet"]["text"])
    
    # Apply preprocessing / predictions pipeline.
    dstream.foreachRDD(lambda x: apply_pipeline(x, model, ss, sc))
    
    # Start Spark Streaming context.
    ssc.start()
    
    ssc.awaitTermination()