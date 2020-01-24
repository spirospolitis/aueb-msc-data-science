import logging
import numpy as np
import pandas as pd

"""
    File: UtilHelper.py
    Date: 01/2020
    Author: Spiros Politis (p3351814)
    Python: 3.6
"""

"""
    Data Mining Homework 1.
"""


class UtilHelper:

    """
        Constructor
    """
    def __init__(self):
        # Define global logging parameters.
        logging.basicConfig(format = "%(levelname)s: %(message)s", level = logging.DEBUG, datefmt = "%I:%M:%S")
        
        # Get instance of log.
        self._log = logging.getLogger()

    """

        :param df: Pandas DataFrame. 
        :param column: Pandas DataFrame column name.
    """
    def check_na(self, df: pd.DataFrame, column: str):
        try:
            assert len(df[df[column].isnull()]) == 0, column + " contains NA values."
            
            self._log.info("{} does not contain NA values.".format(column))
        except AssertionError as ae:
            self._log.warning(ae)

    """
        Provides a set of items. 
        
        :param s: set s to add items to. Set gets mutated at every iteration.
        :param l: list of items to add to set s
    """
    def union(self, s: set, o: object):
        # If type is list iterate over list items.
        if type(o) is list:
            for item in o:
                s.add(item)
        # If any type of object, add it to the set.
        else:
            s.add(o)
            
        return s

    """
        Provides a set of items from a Pandas DataFrame column,
        effectively getting distinct values of the column.
        
        :param df: Pandas DataFrame.
        :param column: Pandas DataFrame column to process.

        :return: set.
    """
    def to_set(self, df: pd.DataFrame, column: str):
        # Set.
        s = set()
        
        # Iterate over a Series object.
        df[column].apply(lambda x: self.union(s, x))
        
        return s
    
    """
        Converts a customer_id to a zero-based index.
        
        :param customer_id: a customer_id.

        :return: Pandas DataFrame index.
    """
    def customer_id_to_index(self, customer_id: np.int32):
        try:
            assert customer_id > 0, "customer_id must be grater than 0"

            return customer_id - 1
        except AssertionError as ae:
            self._log.error(ae)

        return None

    """
        Converts a zero-based index to a customer_id.
        
        :param index: a zero-based Pandas DataFrame index.

        :return: customer id.
    """
    def index_to_customer_id(self, index: np.int32):
        try:
            assert index > 0, "index must be grater than 0"

            return index + 1
        except AssertionError as ae:
            self._log.error(ae)

        return None

    @property
    def log(self):
        return self._log