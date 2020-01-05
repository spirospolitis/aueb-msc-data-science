import logging

import numpy as np
import snap

"""
    File: Algorithms.py
    Date: 11/2019
    Author: Spiros Politis
    Python: 3.6
"""

"""
    SNA Homework 1, Part 2 algorithms.
"""


class Algorithms:

    """
        Constructor
    """
    def __init__(self):
        pass

    """
        Find the ID of the node with the highest degree as well as its degree.
        
        This function iterates over the nodes of the graph and stores [node, degree] pairs in an array.
        The array is the sorted in descending order and its first element is returned.
        
        :param graph: a SNAP graph instance.
        
        :returns: [node, degree] such that degree is max.
    """
    def compute_max_degree(self, graph):
        # Array placeholder.
        arr = []

        # Retrieve node degrees.
        degrees = snap.TIntV()
        snap.GetDegSeqV(graph, degrees)

        # Populate the array.
        for i in range(0, degrees.Len()):
            arr.append([i, degrees[i]])

        # Sort the array.
        arr.sort(key=lambda x: x[1], reverse=True)

        # Return top item.
        return arr[0]

    """
        SNAP reference: https://snap.stanford.edu/snappy/doc/reference/GetHits.html

        :param : a SNAP graph instance.

        :returns: (ordered (desc) hub scores (SNAP THash instance), ordered (desc) authority scores (SNAP THash instance)).
    """
    def compute_hub_authority_score(self, graph):
        # A hash table of int keys and float values (output).
        # The keys are the node ids and the values are the hub scores as outputed by the HITS algorithm.
        # Type: snap.TIntFltH
        hub_scores = snap.TIntFltH()

        # A hash table of int keys and float values (output)
        # The keys are the node ids and the values are the authority scores as outputed by the HITS algorithm.
        # Type: snap.TIntFltH
        authority_scores = snap.TIntFltH()

        snap.GetHits(graph, hub_scores, authority_scores)

        return hub_scores, authority_scores

    """
        SNAP reference: https://snap.stanford.edu/snappy/doc/reference/CommunityGirvanNewman.html

        :param graph: a SNAP graph instance.

        :returns: (modularity metric (int), communities (SNAP THash instance)).
    """
    def compute_girvan_newman(self, graph):
        communities = snap.TCnComV()
        modularity = snap.CommunityGirvanNewman(graph, communities)

        return modularity, communities

    """
        SNAP reference: https://snap.stanford.edu/snappy/doc/reference/CommunityCNM.html

        :param graph: a SNAP graph instance.

        :returns: (modularity metric (int), communities (SNAP THash instance)).
    """
    def compute_clauset_newman_moore(self, graph):
        communities = snap.TCnComV()
        modularity = snap.CommunityCNM(graph, communities)

        return modularity, communities

    """
        SNAP reference: https://snap.stanford.edu/snappy/doc/reference/GetPageRank.html
        
        :param graph: a SNAP graph instance. 
        :param c: damping factor.
        :param eps: convergence rate.
        :param max_iter: max algorithm convergence iterations.

        :returns: ordered (desc) page rank (SNAP THash instance).
    """
    def compute_page_rank(self, graph, c: float = 0.85, eps: float = 10e-4, max_iter: int = 100):
        page_rank = snap.TIntFltH()
        snap.GetPageRank(graph, page_rank, c, eps, max_iter)

        return page_rank

    """
        SNAP reference: https://snap.stanford.edu/snappy/doc/reference/GetBetweennessCentr.html
        
        :param graph: a SNAP graph instance. 
    """
    def compute_betwenness_centrality(self, graph):
        nodes_betweenness_centrality = snap.TIntFltH()
        edges_betweenness_centrality = snap.TIntPrFltH()
        snap.GetBetweennessCentr(graph, nodes_betweenness_centrality, edges_betweenness_centrality, 1.0)

        return nodes_betweenness_centrality, edges_betweenness_centrality

    """
        SNAP reference: https://snap.stanford.edu/snappy/doc/reference/GetClosenessCentr.html
        
        :param graph: a SNAP graph instance. 
    """
    def compute_closeness_centrality(self, graph):
        closeness_centrality = snap.TIntFltH()

        for node in graph.Nodes():
            node_closeness_centrality = snap.GetClosenessCentr(graph, node.GetId())
            closeness_centrality.AddDat(node.GetId(), node_closeness_centrality)

        return closeness_centrality
