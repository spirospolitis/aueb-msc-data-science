import sys
import unittest
import random
import datetime
from timeit import default_timer as timer

import pandas as pd
import snap

from SnaHomework1.Part2 import Algorithms
from SnaHomework1.Part2 import Util

algorithms = Algorithms.Algorithms()
util = Util.Util()

"""
    Although clearly not unit tests, the same approach as "hw1-1.py" was followed.
"""
class TestAlgorithmsMethods(unittest.TestCase):
    # Initial parameters.
    _num_nodes = None
    _out_degree = None
    _rewire_prob = None

    # Parameters of iteration before either memory error occurs or execution time exceeds 10 minutes.
    _last_valid_num_nodes = None
    _last_valid_out_degree = None
    _last_valid_rewire_prob = None

    # Cutoff time: algorithms should take at most 10 minutes to run.
    _cutoff_time = datetime.timedelta(minutes=10)
    # Flag indicating whether we should continue looking for paramaters.
    _running = True
    # Iteration step.
    _iteration = 1

    """
    """
    def test_find_params(self):
        # Initialize parameters.
        self._num_nodes = 50
        self._out_degree = random.randint(5, 20)
        self._rewire_prob = random.random()

        # Run until exit conditions are met.
        while self._running:
            print("-------------------- ITERATION {} --------------------".format(self._iteration))

            # Print current parameters.
            current_parameters_df = util.pretty_print_results([
                ["Number of nodes", int(self._num_nodes)],
                ["Node out degree", int(self._out_degree)],
                ["Node rewire probability", float(self._rewire_prob)]
            ], task="parameters")

            # Generate Watts-Strogatz graph.
            graph = util.generate_watts_strogatz_model(num_nodes=self._num_nodes, out_degree=self._out_degree, rewire_prob=self._rewire_prob)

            ###
            # Task 1: print out the id of the node with the highest degree as well as its degree.
            ##
            max_node_degree=algorithms.compute_max_degree(graph)
            task_1_df = util.pretty_print_results([
                [max_node_degree[0], max_node_degree[1]]
            ], task="max_node_degree")

            ###
            # Task 2: print out the ids of the nodes with the highest Hub and Authority scores as well as their scores.
            ##
            hub_scores, authority_scores = algorithms.compute_hub_authority_score(graph)
            hub_scores = util.TIntFltH_to_array(hub_scores)
            authority_scores = util.TIntFltH_to_array(authority_scores)
            # Sort in descending order, on "score" key.
            hub_scores.sort(key=lambda x: x[1], reverse=True)
            authority_scores.sort(key=lambda x: x[1], reverse=True)
            # Print table.
            task_2_df = util.pretty_print_results([
                [hub_scores[0][0], "Hub", hub_scores[0][1]],
                [authority_scores[0][0], "Authority", authority_scores[0][1]]
            ], task="max_hub_authority_score")

            ###
            # Task 3: measure the time needed for the execution of the Girvan-Newman community detection algorithm based on
            # betweenness centrality and the Clauset-Newman-Moore community detection method.
            ##
            try:
                # Girvan-Newman.
                t1_start = timer()
                algorithms.compute_girvan_newman(graph)
                t1_end = timer()
                t1_elapsed = datetime.timedelta(seconds=(t1_end - t1_start))

                # Clauset-Newman-Moore.
                t2_start = timer()
                algorithms.compute_clauset_newman_moore(graph)
                t2_end = timer()
                t2_elapsed = datetime.timedelta(seconds=(t2_end - t2_start))

                t_total = t1_elapsed + t2_elapsed

                task_3_df = util.pretty_print_results([
                    ["Girvan-Newman", t1_elapsed],
                    ["Clauset-Newman-Moore", t2_elapsed],
                    ["TOTAL", t_total]
                ], task="algorithms_times")

                # Store the current iteration parameters.
                self._last_valid_num_nodes = self._num_nodes
                self._last_valid_out_degree = self._out_degree
                self._last_valid_rewire_prob = self._rewire_prob

                # Advance the next iteration parameters.
                self._num_nodes += 50
                self._out_degree = random.randint(5, 20)
                self._rewire_prob = random.random()
                self._iteration += 1

                # If execution time exceeds 10 minutes.
                if t_total > self._cutoff_time:
                    # Print found parameters.
                    found_parameters_df = util.pretty_print_results([
                        ["Number of nodes", int(self._last_valid_num_nodes)],
                        ["Node out degree", int(self._last_valid_out_degree)],
                        ["Node rewire probability", float(self._last_valid_rewire_prob)]
                    ], task="parameters")

                    self._running = False

            except MemoryError as memory_error:
                # Print found parameters.
                found_parameters_df = util.pretty_print_results([
                    ["Number of nodes", int(self._last_valid_num_nodes)],
                    ["Node out degree", int(self._last_valid_out_degree)],
                    ["Node rewire probability", float(self._last_valid_rewire_prob)]
                ], task="parameters")

                self._running = False

    """
    """
    def test_measures(self):
        # Hard-coded identified parameters.
        NUM_NODES = 450
        OUT_DEGREE = 18
        REWIRE_PROB = 0.987

        # Generate Watts-Strogatz graph.
        graph = util.generate_watts_strogatz_model(num_nodes=NUM_NODES, out_degree=OUT_DEGREE, rewire_prob=REWIRE_PROB)

        ###
        # Task 4: PageRank
        ##
        # Compute PageRank.
        page_rank = algorithms.compute_page_rank(graph)
        # Convert SNAP TIntFltH instance to array.
        page_rank = util.TIntFltH_to_array(page_rank)
        # Print results as a Pandas DataFrame.
        #task_4_df = util.pretty_print_results(page_rank_top_30, task="page_rank")

        ###
        # Task 5: Compute measures.
        ##
        # Betwenenss centrality.
        betweenness_centrality, _ = algorithms.compute_betwenness_centrality(graph)
        # Convert SNAP TIntFltH instance to array.
        betweenness_centrality = util.TIntFltH_to_array(betweenness_centrality)

        # Closeness centrality.
        closeness_centrality = algorithms.compute_closeness_centrality(graph)
        # Convert SNAP TIntFltH instance to array.
        closeness_centrality = util.TIntFltH_to_array(closeness_centrality)

        # Hub and Authority scores.
        hub_scores, authority_scores = algorithms.compute_hub_authority_score(graph)
        # Convert SNAP TIntFltH instances to arrays.
        hub_scores = util.TIntFltH_to_array(hub_scores)
        authority_scores = util.TIntFltH_to_array(authority_scores)

        # Create a Pandas dataframe with all measures.
        df = pd.DataFrame({
            "page_rank": [x[1] for x in page_rank],
            "betweenness_centrality": [x[1] for x in betweenness_centrality],
            "closeness_centrality": [x[1] for x in closeness_centrality],
            "hub_scores": [x[1] for x in hub_scores],
            "authority_scores": [x[1] for x in authority_scores]
        })

        # Sort descending on PageRank.
        df = df.sort_values("page_rank", ascending=False)

        # Slice top-30 values.
        df = df.head(30)

        # Show the dataframe.
        print(df)

        ###
        # Produce plots.
        ##
        util.plot_page_rank_vs_betweenness_closeness(df)
        util.plot_page_rank_vs_authority_hub(df)

if __name__ == "__main__":
    unittest.main()
