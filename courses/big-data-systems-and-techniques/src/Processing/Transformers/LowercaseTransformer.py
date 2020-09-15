import pyspark.sql
import pyspark.sql.types
import pyspark.ml
import pyspark.sql.functions

"""
    File: LowercaseTransformer.py
    Date: 07/2020
    Author: Spiros Politis
    Python: 3.6
"""

"""
    A custom pipeline stage (Transformer) that converts a string column to lowercase.
"""
class LowercaseTransformer(pyspark.ml.Transformer):
    def __init__(
        self, 
        column = None
    ):
        super(LowercaseTransformer, self).__init__()
        
        self.__column = column
        
    def _transform(self, df):
        def f(s):
            tokens = s.split()
            return " ".join([t.lower() for t in tokens])

        t = pyspark.sql.types.StringType()
        
        return df.withColumn(self.__column, pyspark.sql.functions.udf(f, t)(df[self.__column]))