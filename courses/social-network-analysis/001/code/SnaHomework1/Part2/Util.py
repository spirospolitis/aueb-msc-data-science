import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import snap

"""
    File: Util.py
    Date: 11/2019
    Author: Spiros Politis
    Python: 3.6
"""

"""
    SNA Homework 1, Part 2 utility functions.
"""


class Util:

    """
        Constructor
    """
    def __init__(self):
        pass

    '''
        Convenience function to convert a set of key-value pairs 
        from a snap.TIntFltH type to array.

        :param : a SNAP hashtable (snap.THash)

        :returns: array.
    '''
    def TIntFltH_to_array(self, TIntFltH: snap.TIntFltH):
        arr = []

        for key in TIntFltH:
            arr.append([key, TIntFltH[key]])

        return arr

    """
        Generate graphs using the Watts-Strogatz model.
        
        SNAP reference: https://snap.stanford.edu/snappy/doc/reference/GenSmallWorld.html
    """
    def generate_watts_strogatz_model(self, num_nodes: int = 10, out_degree: int = None, rewire_prob: float = 0.5):
        if out_degree is None:
            # Generate a random out degree in [5, 20].
            out_degree = random.randint(5, 20)

        rnd = snap.TRnd(1, 0)
        graph = snap.GenSmallWorld(num_nodes, out_degree, rewire_prob, rnd)

        return graph

    """
    """
    def pretty_print_results(self, data: pd.DataFrame, task: str):
        # Create Pandas data frame.
        if task == "parameters":
            df = pd.DataFrame(data, columns=["Parameter", "Value"])
            text = "Execution parameters"

        if task == "max_node_degree":
            df = pd.DataFrame(data, columns=["Node ID", "Degree"])
            text = "Node with highest degree"

        if task == "max_hub_authority_score":
            df = pd.DataFrame(data, columns=["Node ID", "Type", "Score"])
            text = "IDs of nodes with the highest Hub and Authority scores, along with their scores"

        if task == "algorithms_times":
            df = pd.DataFrame(data, columns=["Algorithm", "Time"])
            text = "Execution times of the Girvan-Newman community detection algorithm and the Clauset-Newman-Moore community detection method"

        if task == "page_rank":
            df = pd.DataFrame(data, columns=["Node ID", "PageRank"])
            text = "PageRank table"

        if task == "betweenness_centrality":
            df = pd.DataFrame(data, columns=["Node ID", "Betweenness_centrality"])
            text = "Betweenness centrality table"

        if task == "closeness_centrality":
            df = pd.DataFrame(data, columns=["Node ID", "Closeness centrality"])
            text = "Closeness centrality table"

        if task == "hub":
            df = pd.DataFrame(data, columns=["Node ID", "Hub score"])
            text = "Hub scores table"

        if task == "authority":
            df = pd.DataFrame(data, columns=["Node ID", "Authority score"])
            text = "Authority scores table"

        with pd.option_context("display.float_format", "{:0.20f}".format):
            print("\n")
            print(text)
            print("---------------------------------------------------")
            print(df)
            print("---------------------------------------------------")
            print("\n")

        return df

    """
    """
    def plot_page_rank_vs_betweenness_closeness(self, df: pd.DataFrame):
        plot_main_title_font = {
            "family": "sans serif",
            "color": "black",
            "weight": "bold",
            "size": 18
        }

        fig = plt.figure(figsize=(12, 8))

        # PageRank axis.
        ax1 = fig.add_subplot(111)
        # Betwenness axis.
        ax2 = ax1.twinx()
        # Closeness axis.
        ax3 = ax1.twinx()

        # Bar width.
        width = 0.1

        # Bar plots.
        plot_1 = df.page_rank.plot(kind="bar", color="tab:blue", ax=ax1, width=width, position=2, label="PageRank")
        plot_2 = df.betweenness_centrality.plot(kind="bar", color="tab:olive", ax=ax2, width=width, position=1, label="Betweenness centrality")
        plot_3 = df.closeness_centrality.plot(kind="bar", color="tab:grey", ax=ax3, width=width, position=0, label="Closeness centrality")

        # Plot title.
        ax1.set_title(label="PageRank vs. Betwenness, Closeness centrality", loc="center", fontdict=plot_main_title_font, pad=20)

        # Axis labels.
        ax1.set_ylabel("PageRank", labelpad=10)
        ax1.set_xlabel("Node ID", labelpad=10)
        ax1.tick_params(axis="y", colors="tab:blue")
        ax1.yaxis.label.set_color("tab:blue")

        ax2.set_ylabel("Betweeness centrality", labelpad=10)
        ax2.tick_params(axis="y", colors="tab:olive")
        ax2.yaxis.label.set_color("tab:olive")

        ax3.set_ylabel("Closeness centrality", labelpad=30)
        ax3.tick_params(axis="y", colors="tab:grey")
        ax3.yaxis.label.set_color("tab:grey")

        # Save the figure.
        fig.savefig("figures/page_rank_vs_betwenness_closenes.png")

        plt.show()

    """
    """
    def plot_page_rank_vs_authority_hub(self, df: pd.DataFrame):
        plot_main_title_font = {
            "family": "sans serif",
            "color": "black",
            "weight": "bold",
            "size": 18
        }

        fig = plt.figure(figsize=(12, 8))

        # PageRank axis.
        ax1 = fig.add_subplot(111)
        # Betwenness axis.
        ax2 = ax1.twinx()
        # Closeness axis.
        ax3 = ax1.twinx()

        # Bar width.
        width = 0.1

        # Bar plots.
        plot_1 = df.page_rank.plot(kind="bar", color="tab:blue", ax=ax1, width=width, position=2, label="PageRank")
        plot_2 = df.hub_scores.plot(kind="bar", color="tab:olive", ax=ax2, width=width, position=1, label="Hub score")
        plot_3 = df.authority_scores.plot(kind="bar", color="tab:grey", ax=ax3, width=width, position=0, label="Authority score")

        # Plot title.
        ax1.set_title(label="PageRank vs. Hub, Authority scores", loc="center", fontdict=plot_main_title_font, pad=20)

        # Axis labels.
        ax1.set_ylabel("PageRank", labelpad=10)
        ax1.set_xlabel("Node ID", labelpad=10)
        ax1.tick_params(axis="y", colors="tab:blue")
        ax1.yaxis.label.set_color("tab:blue")

        ax2.set_ylabel("Hub score", labelpad=10)
        ax2.tick_params(axis="y", colors="tab:olive")
        ax2.yaxis.label.set_color("tab:olive")

        ax3.set_ylabel("Authority score", labelpad=30)
        ax3.tick_params(axis="y", colors="tab:grey")
        ax3.yaxis.label.set_color("tab:grey")

        # Save the figure.
        fig.savefig("figures/page_rank_vs_hub_authority.png")

        plt.show()