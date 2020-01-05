import random

import snap

from .Util import Util
from .GraphEvaluator import GraphEvaluator

"""
    File: GraphGenerator.py
    Date: 11/2019
    Author: Spiros Politis
    Python: 3.6
"""

"""
    SNA Homework 1, Part 1 graph generator functions.
"""


class GraphGenerator:
    """
        Constructor
    """

    def __init__(self):
        self.graph_evaluator = GraphEvaluator()
        self.util = Util()

    """
        Recursively creates a connected graph.
        
        :param graph: SNAP graph instance.
        :param node_a: a node in the graph. Used in recursion as a starting point. 
        :param node_b: a node in the graph. Used in recursion as a starting point.
        :param connected_nodes: list of recursively connected nodes.  
        
        :returns: SNAP graph instance.
    """
    def create_connected_graph(self, graph, node_a=None, node_b=None, connected_nodes=None):
        # Initialize the connected nodes array.
        if connected_nodes is None:
            connected_nodes = []

        # Initial conditions.
        # Pick a couple of random nodes to connect, initially.
        if node_a is None and node_b is None:
            node_a = random.randrange(graph.GetNodes())
            node_b = None

            # Loop while node_b is not the same as node_a.
            while node_b is None or node_a == node_b:
                node_b = random.randrange(graph.GetNodes())

            # Add edge (node_a, node_b).
            graph.AddEdge(node_a, node_b)

            # Add nodes to list of connected nodes.
            connected_nodes.append(node_a)
            connected_nodes.append(node_b)

        # Nodes have been connected from previous iteration.
        else:
            # Pick a random node from the set of {node_a, node_b} as the initial node for the next edge.
            # If false, set node_a as node_b.
            if not bool(random.getrandbits(1)):
                node_a = node_b
                node_b = None
                while node_b is None or node_b in connected_nodes:
                    node_b = random.randrange(graph.GetNodes())

                # Add edge (node_a, node_b).
                graph.AddEdge(node_a, node_b)

                # Add node to list of connected nodes.
                connected_nodes.append(node_b)
            # Otherwise select node_a and pick another candidate node for connection.
            else:
                node_b = None
                while node_b is None or node_b in connected_nodes:
                    node_b = random.randrange(graph.GetNodes())

                # Add edge (node_a, node_b).
                graph.AddEdge(node_a, node_b)

                # Add node to list of connected nodes.
                connected_nodes.append(node_b)

        # Recursion exit condition.
        # If graph is connected, return the graph.
        if snap.IsConnected(graph):
            return graph
        # If not, continue until a connected graph is created.
        else:
            return self.create_connected_graph(graph, node_a, node_b, connected_nodes)

    """
        Generates a Eulerian graph.
        
        :param graph: a Snap graph instance.

        :returns: a Snap graph instance.
    """
    def create_euler_graph(self, graph):
        # Get graph degree table.
        nodes_degrees = self.util.get_in_out_degree_table(graph)

        # Get odd degree nodes length.
        odd_degree_count = nodes_degrees[nodes_degrees[:, 1] % 2 != 0].shape[0]

        # If the graph has exactly 2 odd degree nodes.
        if odd_degree_count == 2:
            # If an edge exists between node_a and node_b, delete it.
            node_a = int(nodes_degrees[nodes_degrees[:, 1] % 2 != 0][0, 0])
            node_b = int(nodes_degrees[nodes_degrees[:, 1] % 2 != 0][1, 0])

            # if graph.IsEdge(node_a, node_b):
            #     graph.DelEdge(node_a, node_b)

            return graph
        else:
            # Pick two random nodes with odd degree.
            # Function random.sample picks from a set without replacement.
            node_a_b = random.sample(set(nodes_degrees[nodes_degrees[:, 1] % 2 != 0][:, 0]), 2)

            graph.AddEdge(int(node_a_b[0]), int(node_a_b[1]))

            # Recurse.
            return self.create_euler_graph(graph)

    """
        Generates a graph that has an Euler path (but not an Euler circuit).
        
        :param num_nodes: number of graph nodes.
        
        :returns: SNAP graph instance.
    """
    def generate_has_euler_path_but_not_circuit(self, num_nodes: int = 10):
        # Generate an Erdos-Renyi random graph with num_nodes and zero edges.
        graph = snap.GenRndGnm(snap.PUNGraph, num_nodes, 0)

        # Create a connected graph.
        graph = self.create_connected_graph(graph)

        # Create a graph with an Euler path.
        graph = self.create_euler_graph(graph)

        return graph

    """
        Generates a graph that does not have an Euler path.
        
        :param num_nodes: number of graph nodes.
        
        :returns: SNAP graph instance.
    """
    def generate_does_not_have_euler_path(self, num_nodes: int = 10):
        # Since a graph has an Euler path if and only if there are at most two vertices with odd degree,
        # we must make sure that we augment the odd degree by one.

        # First, create a graph with an Euler path.
        graph = self.generate_has_euler_path_but_not_circuit(num_nodes)

        # Get graph degree table.
        nodes_degrees = self.util.get_in_out_degree_table(graph)

        # Retrieve a random vertex with even degree.
        even_degree_node = random.sample(set(nodes_degrees[nodes_degrees[:, 1] % 2 == 0][:, 0]), 1)

        # Add an extra vertex and edge connecting them.
        graph.AddNode(num_nodes)
        graph.AddEdge(int(even_degree_node[0]), num_nodes)

        return graph

    """
        Generates a graph that has an Euler circuit.
        
        :param num_nodes: number of graph nodes.
        
        :returns: SNAP graph instance.
    """
    def generate_has_euler_circuit(self, num_nodes: int = 10):
        # Generate a graph that has an Euler path but no circuit.
        graph = self.generate_has_euler_path_but_not_circuit(num_nodes)

        # Add an edge between start and end node to create a Euler circuit.
        # Get start and end nodes.
        _, vertices = self.graph_evaluator.has_euler_path(graph)
        # Connect them.
        graph.AddEdge(int(list(vertices)[0]), int(list(vertices)[1]))

        return graph

    """
        Generates a graph that does not have an Euler circuit.
        
        :param num_nodes: number of graph nodes.
        
        :returns: SNAP graph instance.
    """
    def generate_does_not_have_euler_circuit(self, num_nodes:int = 10):
        # Generate a graph with a Euler circuit.
        graph = self.generate_has_euler_circuit(num_nodes)

        # One way to guarantee that a graph does not have an Euler circuit is to include a "spike",
        # that is a vertex of degree 1.
        # Add a single node and an edge to convert it to a non-Eulerian graph.
        graph.AddNode(num_nodes)
        # The edge must connect the start-end of the Euler path to the new vertex.
        graph.AddEdge(0, num_nodes)

        return graph
