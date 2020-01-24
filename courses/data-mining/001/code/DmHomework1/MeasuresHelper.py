import numpy as np
import pandas as pd

from .UtilHelper import UtilHelper

"""
    File: MeasuresHelper.py
    Date: 01/2020
    Author: Spiros Politis (p3351814)
    Python: 3.6
"""

"""
    Data Mining Homework 1.
"""


class MeasuresHelper:

    """
        Constructor
    """
    def __init__(self):
        self._util_helper = UtilHelper()

        # Dictionary of variables / metrics to be applied for computing dissimilarity.
        # The dictionary indicates the appropriate class function to call for every variable type.
        self._dissimilarity_measures_def = {
            "age": "numeric_dissimilarity",
            "job": "nominal_dissimilarity", 
            "marital": "nominal_dissimilarity", 
            "education": "ordinal_dissimilarity",
            "default": "nominal_dissimilarity", 
            "balance": "numeric_dissimilarity", 
            "housing": "nominal_dissimilarity", 
            "loan": "nominal_dissimilarity", 
            "products": "set_dissimilarity"
        }

    """
        Computes a vector of dissimilarities for a specific dataframe column
        whose values are of numeric semantics.
        
        :param df: Pandas DataFrame
        :param column: Pandas DataFrame column name
        :param index_to_compare: the row index of the column to compare.
        
        :return: column vector (numpy.Series) of numeric dissimilarities.
    """
    def numeric_dissimilarity(self, df: pd.DataFrame, column: str, index_to_compare: np.int32):
        self._util_helper.log.debug("computing numeric dissimilarity for {}".format(column))

        min_value = np.min(df[column])
        max_value = np.max(df[column])
        
        return np.abs(df.loc[index_to_compare, column] - df[column]) / (max_value - min_value)

    """
        Computes a vector of dissimilarities for a specific dataframe column
        whose values are of nominal semantics.
        
        :param df: Pandas DataFrame
        :param column: Pandas DataFrame column name
        :param index_to_compare: the row index of the column to compare.
        
        :return: column vector (numpy.Series) of nominal dissimilarities.
    """
    def nominal_dissimilarity(self, df: pd.DataFrame, column: str, index_to_compare: np.int32):
        self._util_helper.log.debug("computing nominal dissimilarity for {}".format(column))

        return pd.Series(np.array(df.loc[index_to_compare, column] != df[column], dtype = np.float64))

    """
        Computes a vector of dissimilarities for a specific dataframe column
        whose values are of cardinal semantics.
        
        :param df: Pandas DataFrame
        :param column: Pandas DataFrame column name
        :param index_to_compare: the row index of the column to compare.
        
        :return: column vector (numpy.Series) of cardinal dissimilarities.
    """
    def ordinal_dissimilarity(self, df: pd.DataFrame, column: str, index_to_compare: np.int32):
        self._util_helper.log.debug("computing ordinal dissimilarity for {}".format(column))

        min_rank = np.min(df[column])
        max_rank = np.max(df[column])
        
        return np.abs(df.loc[index_to_compare, column] - df[column]) / (max_rank - min_rank)

    """
        Computes the Jaccard index vector.

        :param df: Pandas DataFrame
        :param column: Pandas DataFrame column name
        :param index_to_compare: the row index of the column to compare.
        
        :return: column vector (numpy.Series) of Jaccard index values.
    """
    def set_dissimilarity(self, df: pd.DataFrame, column: str, index_to_compare: np.int32):
        self._util_helper.log.debug("computing set dissimilarity for {}".format(column))

        # Define a vector to hold Jaccard index values.
        jaccard_index_vector = np.zeros(len(df), dtype = np.float64)

        # Iterate over rows in the DataFrame.
        for index, row in df.iterrows():
            # Compute intersection set of current products list with products list of comparing item. 
            intersection_set = set(df.loc[index_to_compare, column]) & set(row[column])

            # Compute union set of current products list with products list of comparing item.
            union_set = set(df.loc[index_to_compare, column]) | set(row[column])
            
            intersection_set_len = len(intersection_set)
            union_set_len = len(union_set)

            # Compute the Jaccard value.
            jaccard_index_vector[index] = 1 - (intersection_set_len / union_set_len)

        return pd.Series(jaccard_index_vector)

    """
        Computes all measures based on metric type (similarity or dissimilarity).

        :param df: Pandas DataFrame
        :param index_to_compare: the row index of the column to compare.
        
        :return: matrix of metrics.
    """
    def all(self, df: pd.DataFrame, index_to_compare: np.int32):
        self._util_helper.log.debug("computing all dissimilarity measures")

        matrix = np.zeros((len(df), len(self._dissimilarity_measures_def.keys())), dtype = np.float64)

        for i, (key, _) in enumerate(self._dissimilarity_measures_def.items()):
            matrix[:, i] = self.__getattribute__(self._dissimilarity_measures_def[key])(df = df, column = key, index_to_compare = index_to_compare)

        # Calculate mean dissimilarity per row (customer).
        matrix = np.hstack((matrix, np.mean(matrix, axis = 1).reshape(len(df), 1)))

        return matrix

    """
        Computes n nearest neighbors of a given customer.

        :param df: Pandas DataFrame.
        :param index_to_compare: index of the customer to compare against all others.
        :param n_neighbors: number of neighbors.
        
        :return: nearest neighbors (value, index) arrays tuple. 
        First array contains the similarity values, second array is the indices in the original dataframe (customers).
    """
    def nn(self, matrix: np.ndarray, n_neighbors: np.int32):
        # Sort by mean dissimilarity, ascending, get top-10 excluding first (reference customer)
        n_most_similar_indices = matrix[:, 9].argsort()[1:n_neighbors]
        n_most_similar_customers = map(self._util_helper.index_to_customer_id, list(n_most_similar_indices))

        return n_most_similar_indices, list(n_most_similar_customers)

    """
        Computes the requirements of the assignment.

        :param df: Pandas DataFrame.
        :param customer_id: the customer ID against which to compute similarity.

        :return: list of top-10 customers, by similarity to customer_id.
    """
    def get_most_similar_customers(self, df: pd.DataFrame, customer_id: np.int32):
        matrix = self.all(df = df, index_to_compare = self._util_helper.customer_id_to_index(customer_id))

        _,  n_most_similar_customers = self.nn(matrix = matrix, n_neighbors = 11)

        return n_most_similar_customers
