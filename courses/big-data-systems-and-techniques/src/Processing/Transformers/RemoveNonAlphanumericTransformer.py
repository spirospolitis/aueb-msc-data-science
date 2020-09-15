import pyspark.sql
import pyspark.sql.types
import pyspark.ml
import pyspark.sql.functions
import re

"""
    File: RemoveNonAlphanumericTransformer.py
    Date: 07/2020
    Author: Spiros Politis
    Python: 3.6
"""

"""
    A custom pipeline stage (Transformer) that removes punctuation 
    and excessive whitespace characters from a string column.
"""
class RemoveNonAlphanumericTransformer(pyspark.ml.Transformer):
    def __init__(
        self, 
        column = None
    ):
        super(RemoveNonAlphanumericTransformer, self).__init__()
        
        self.__column = column
        
    def _transform(self, df):
        def f(s):
            # Remove non-alphanumeric characters.
            s = re.sub("\W", " ", s)
        
            # Remove excessive white space characters.
            s = re.sub(" +", " ", s)
            
            return s
        
        t = pyspark.sql.types.StringType()
        
        return df.withColumn(self.__column, pyspark.sql.functions.udf(f, t)(df[self.__column]))