import snap

from .Util import Util

"""
    File: GraphEvaluator.py
    Date: 11/2019
    Author: Spiros Politis
    Python: 3.6
"""

"""
    SNA Homework 1, Part 1 graph generator functions.
"""


class GraphEvaluator:

    """
        Constructor
    """
    def __init__(self):
        self.util = Util()

    """
        An Euler path is a path that uses every edge of a graph exactly once.
    
        An Euler path starts and ends at different vertices.
    
        If a graph has an Euler path, then it must have exactly two vertices with odd degree, 
        and it is these odd vertices that will form the beginning and end of the path.
    
        :param graph: a Snap graph instance.
    
        :returns: a Snap graph instance.
    """
    def has_euler_path(self, graph):
        vertices = set()
        odd_degree_count = 0

        ###
        # Condition: verify that the graph is connected.
        #
        # If not, the graph cannot have an Euler path.
        ##
        if snap.IsConnected(graph) == False:
            # print("has_euler_path()::disconnected graph")

            return False, vertices

        # Type snap.PUNGraph is of an undirected graph.
        if type(graph) is snap.PUNGraph:
            nodes_degrees = self.util.get_in_out_degree_table(graph)

            # Iterate table of node degrees.
            for i in range(nodes_degrees.shape[0]):

                # If node has odd degree.
                if nodes_degrees[i, 1] % 2 != 0:
                    # Add the vertex with odd degree to the vertices set.
                    vertices.add(nodes_degrees[i, 0])

                    # Increase the odd degree counter by 1.
                    odd_degree_count += 1

                # Loop exit condition, if odd degree count surpasses 2.
                if odd_degree_count > 2:
                    # print("has_euler_path()::graph has odd degree > 2")

                    return False, set()

            if odd_degree_count != 2:
                # print("has_euler_path()::graph has odd degree != 2 ({})".format(odd_degree_count))

                return False, set()
            else:
                # print("has_euler_path()::found Euler path")

                return True, vertices

        # Type snap.PNGraph is of a directed graph.
        if type(graph) is snap.PNGraph:
            pass

        return False, set()

    """
        :param graph: a Snap graph instance.
    
        :returns: a Snap graph instance.
    """
    def has_euler_circuit(self, graph):
        ###
        # Condition: verify that the graph is connected.
        ##
        if snap.IsConnected(graph) == False:
            return False

        # Type snap.PUNGraph is of an undirected graph.
        if type(graph) is snap.PUNGraph:
            nodes_degrees = self.util.get_in_out_degree_table(graph)

            # Iterate table of node degrees.
            for i in range(nodes_degrees.shape[0]):

                # If a node with odd degree is found, return False.
                if nodes_degrees[i, 1] % 2 != 0:
                    return False

            return True

        # Type snap.PNGraph is of a directed graph.
        if type(graph) is snap.PNGraph:
            pass