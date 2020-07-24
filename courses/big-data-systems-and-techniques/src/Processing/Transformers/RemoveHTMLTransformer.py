import pyspark.sql
import pyspark.sql.types
import pyspark.ml
import pyspark.sql.functions
import os

"""
    File: AbstractStorage.py
    Date: 07/2020
    Author: Spiros Politis
    Python: 3.6
"""

"""
    A custom pipeline stage (Transformer) that removes HTML from a string column.
"""
class RemoveHTMLTransformer(pyspark.ml.Transformer):
    def __init__(
        self, 
        column, 
        sc
    ):
        super(RemoveHTMLTransformer, self).__init__()

        self.__column = column
        self.__sc = sc

        self.__install_beautifulsoup4_package()
        
    
    def __install_beautifulsoup4_package(self):
        def beautifulsoup4(x):
            os.system("pip install beautifulsoup4")

            return x

        # Two worker nodes.
        rdd = self.__sc.parallelize([1, 2])

        rdd.map(lambda x: beautifulsoup4(x)).collect()
        
    def _transform(self, df):
        from bs4 import BeautifulSoup
        
        def f(s):    
            return "".join(p for p in BeautifulSoup(s, features = "html.parser").get_text())

        t = pyspark.sql.types.StringType()
        
        return df.withColumn(self.__column, pyspark.sql.functions.udf(f, t)(df[self.__column]))