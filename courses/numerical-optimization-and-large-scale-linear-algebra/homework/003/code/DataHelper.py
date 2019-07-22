'''
    AUEB M.Sc. in Data Science (part-time)
    Course: Numerical Optimization and Large Scale Linear Algebra
    Semester: Spring 2019
    Instructor: P. Vassalos
    Student: S. Politis
    Student ID: p3351814
    Date: 18/06/2019

    Homework 3
'''

import numpy as np

'''
    Implements data generation functions for the homework.
'''
class DataHelper():

    '''
        Constructor
    '''
    def __init__(self):
        self.__dataframe = None
        self.__graph = None
        self.__sparse_row_matrix = None


    '''
        Ingests the data file of matrices A, B, G, returns the matrices.

        :param file_path: File path.

        :returns: 
    '''
    def ingest_data(self, file_path:str):
        import numpy as np
        import pandas as pd

        data_df = pd.read_csv(file_path, sep = '\t', header = None, dtype = np.float64)
        data_df[0] = data_df[0].astype(np.int32)
        data_df[1] = data_df[1].astype(np.int32)

        # Create the dataframe structure.
        self.__dataframe = data_df

        # Create the graph structure.
        self.__graph = self.__as_graph()

        # Create the Scipy CSR matrix.
        self.__sparse_row_matrix = self.__as_sparse_row_matrix()

        return self



    '''
    '''
    def __as_graph(self):
        import numpy as np
        import networkx as nx

        # Get the dataframe values as a matrix.
        m = self.__dataframe.values

        # Instantiate the graph object.
        # It is a directed graph since the links have direction (from page - to page).
        graph = nx.DiGraph(name = 'Google matrix graph')

        for i in range(0, m.shape[0]):
            # Add graph nodes and edges.
            graph.add_node(m[i, 0].astype(np.int32))
            graph.add_node(m[i, 1].astype(np.int32))
            graph.add_edge(m[i, 0].astype(np.int32), m[i, 1].astype(np.int32), weight = m[i, 2].astype(np.float64))

        return graph



    '''
    '''
    def __as_sparse_row_matrix(self):
        import numpy as np
        import networkx as nx

        sparse_row_matrix = nx.to_scipy_sparse_matrix(self.__graph, np.sort(self.__graph.nodes), format = 'csr', dtype = np.float64)

        return sparse_row_matrix



    '''
    '''
    def add_page(self, node:np.int32, inlinks:np.ndarray = None, outlinks:np.ndarray = None):
        import numpy as np
        import networkx as nx
        
        # Add the node to the graph.
        self.__graph.add_node(node)

        # If page has incoming links, create appropriate edges.
        if inlinks is not None:
            for i in range(0, inlinks.shape[0]):
                self.__graph.add_edge(inlinks[i, 0].astype(np.int32), node, weight = inlinks[i, 1].astype(np.float64))
        
        # If page has outgoing links, create appropriate edges.
        if outlinks is not None:
            for i in range(0, outlinks.shape[0]):
                self.__graph.add_edge(node, outlinks[i, 0].astype(np.int32), weight = outlinks[i, 1].astype(np.float64))
        
        # Re-create the Scipy CSR matrix, since the graph has changed.
        self.__sparse_row_matrix = self.__as_sparse_row_matrix()

        # When a webpage has no outgoing links, we add a 1 as the corresponding 
        # diagonal element of P for making its row-sum one.
        if outlinks is None:
            self.__sparse_row_matrix[node - 1, node - 1] = 1.0

        return self



    '''
        Saves a Numpy array to disk.

        :param : 
        :param : 
    '''
    def np_save(self, filepath:str, a:np.ndarray, fmt:str = 'bin'):
        import numpy as np

        # Check params.
        if fmt not in ['bin', 'ascii']:
            raise ValueError("fmt is either 'bin' or 'ascii'")

        if fmt == 'bin':
            np.save(filepath, a)

        if fmt == 'ascii':
            np.savetxt(filepath, a, fmt = '%.16f')



    '''
        Loads a Numpy array from disk.

        :param :

        :returns: Numpy array.
    '''
    def np_load(self, filepath:str, fmt:str = 'bin'):
        import numpy as np

        # Check params.
        if fmt not in ['bin', 'ascii']:
            raise ValueError("fmt is either 'bin' or 'ascii'")

        if fmt == 'bin':
            return np.load(filepath)

        if fmt == 'ascii':
            return np.loadtxt(filepath, dtype = np.float64)



    '''
        Class properties.
    '''
    @property
    def dataframe(self):
        return self.__dataframe

    @property
    def sparse_row_matrix(self):
        return self.__sparse_row_matrix

    @property
    def graph(self):
        return self.__graph