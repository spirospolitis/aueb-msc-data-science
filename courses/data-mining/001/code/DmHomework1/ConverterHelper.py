import numpy as np
import pandas as pd
import pandas.api.types as pdt

"""
    File: ConverterHelper.py
    Date: 01/2020
    Author: Spiros Politis (p3351814)
    Python: 3.6
"""

"""
    Data Mining Homework 1.
"""


class ConverterHelper:

    """
        Constructor
    """
    def __init__(self):
        pass

    """

        :param : 
        :param : 
        :param : 

        :return: Pandas DataFrame.
    """
    def nominal_to_numeric(self, df: pd.DataFrame, column: str, create_column: bool = False):
        df[column] = pd.Categorical(df[column])
        
        if create_column:
            df[column + "_numeric"] = df[column].cat.codes + 1
        else:
            df[column] = df[column].cat.codes + 1
        
        return df

    """

        :param : 
        :param : 
        :param : 

        :return: Pandas DataFrame.
    """
    def ordinal_to_numeric(self, df: pd.DataFrame, column: str, categories: pd.Series, create_column: bool = False):
        column_categories = pdt.CategoricalDtype(
            categories = categories,
            ordered = True
        )
        
        df[column] = df[column].astype(column_categories)

        if create_column:
            df[column + "_numeric"] = df[column].cat.codes + 1
        else:
            df[column] = df[column].cat.codes + 1

        return df

    """

        :param : 
        :param : 
        :param : 

        :return: Pandas DataFrame.
    """
    def comma_separated_to_list(self, df: pd.DataFrame, column: str, create_column: bool = False):
        if create_column:
            df[column + "_list"] = df[column].apply(lambda x: str(x).split(","))
        else:
            df[column] = df[column].apply(lambda x: str(x).split(","))

        return df

