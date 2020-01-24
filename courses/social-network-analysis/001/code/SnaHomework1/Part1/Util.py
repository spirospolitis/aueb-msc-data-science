import numpy as np
import snap

"""
    File: Util.py
    Date: 11/2019
    Author: Spiros Politis
    Python: 3.6
"""

"""
    SNA Homework 1, Part 1 utility functions.
"""


class Util:

    """
        Constructor
    """
    def __init__(self):
        pass

    """
        Returns a numpy 2D array of shape (graph.GetNodes(), 4) in which:

        - Index is the node number
        - Col 0 is the node name (number)
        - Col 1 is the node in degree
        - Col 2 is the node out degree

        :param graph: a Snap graph instance.

        :returns: Numpy array
    """

    def get_in_out_degree_table(self, graph):
        # Placeholder for node / degree / out degree.
        nodes_degrees = np.zeros((graph.GetNodes(), 3), dtype=np.int32)

        # In degree vector.
        in_degree_v = snap.TIntPrV()
        snap.GetNodeInDegV(graph, in_degree_v)

        # Out degree vector.
        out_degree_v = snap.TIntPrV()
        snap.GetNodeOutDegV(graph, out_degree_v)

        # Set the nodes_degrees Numpy array.
        for item in in_degree_v:
            node = item.GetVal1()
            nodes_degrees[node, 0] = node
            nodes_degrees[node, 1] = item.GetVal2()

        for item in out_degree_v:
            node = item.GetVal1()
            # nodes_degrees[node, 0] = node
            nodes_degrees[node, 2] = item.GetVal2()

        return nodes_degrees