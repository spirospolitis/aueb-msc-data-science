'''
    AUEB M.Sc. in Data Science (part-time)
    Course: Numerical Optimization and Large Scale Linear Algebra
    Semester: Spring 2019
    Instructor: P. Vassalos
    Student: S. Politis
    Student ID: p3351814
    Date: 27/06/2019

    Homework 3
'''

import numpy as np

'''
    Implements visualization functions for the homework.
'''
class VisualizationHelper():

    def plot_graph(self, graph, highlight_nodes:[] = None, size = (10, 10), title:str = None):
        import numpy as np
        import matplotlib.pyplot as plt
        import networkx as nx
        
        plot_main_title_font = {
            'family': 'sans serif',
            'color':  'black',
            'weight': 'bold',
            'size': 18
        }

        _, ax = plt.subplots(figsize = size)
        
        # Set title.
        if title != None:
            ax.set_title(label = title, loc = 'center', fontdict = plot_main_title_font, pad = 20)
        
        position = nx.shell_layout(graph)

        out_edges = [(u, v) for (u, v, d) in graph.out_edges(data = True)]

        out_edge_labels = dict([((u, v, ), '{:.5f}'.format(d['weight'])) for u, v, d in graph.out_edges(data = True)])

        # Highlight some nodes.
        node_color_map = []

        if highlight_nodes is not None:
            for node in graph:
                if node in highlight_nodes:
                    node_color_map.append('orange')
                else: node_color_map.append('grey')
        else:
            node_color_map = ['grey']

        # Draw nodes.
        nx.draw_networkx_nodes(
            graph, 
            pos = position, 
            node_color = node_color_map,
            #node_color = 'orange', 
            node_size = 500, 
            alpha = 0.5, 
            ax = ax
        )
        
        # Draw node labels.
        nx.draw_networkx_labels(
            graph,
            pos = position,
            font_weight = 'bold',
            ax = ax
        )
        
        # Draw out-edges.
        nx.draw_networkx_edges(
            graph,
            pos = position,
            edgelist = out_edges,
            width = 2,
            alpha = 0.5,
            edge_color = 'black',
            ax = ax
        )

        # Draw out-edge labes.
        nx.draw_networkx_edge_labels(
            graph,
            edgelist = out_edges, 
            pos = position, 
            edge_labels = out_edge_labels, 
            label_pos = 0.8, 
            font_size = 9, 
            alpha = 0.8, 
            ax = ax
        )

        return ax



    '''
        Point plot of iterations versus error.

        :param stats: Matrix of statistics from algorithm runs.
        :param size: Plot size.
        :param title: Plot title. 

        :returns: Matplotlib axis object.
    '''
    def plot_iterations_vs_error(self, stats:np.ndarray, size = (10, 10), title:str = None):
        import numpy as np
        import matplotlib.pyplot as plt
        
        plot_main_title_font = {
            'family': 'sans serif',
            'color':  'black',
            'weight': 'bold',
            'size': 18
        }

        _, ax = plt.subplots(figsize = size)
        
        # Set title.
        if title != None:
            ax.set_title(label = title, loc = 'center', fontdict = plot_main_title_font, pad = 20)

        ax.grid(b = True, which = 'both', axis = 'both')

        # Plot
        ax.scatter(x = stats[:, 0], y = stats[:, 1], c = 'blue', marker = '.', s = 50, alpha = 0.7)

        return ax

    
    '''
        Point plot of iterations versus converged ranks.

        :param stats: Matrix of statistics from algorithm runs.
        :param size: Plot size.
        :param title: Plot title. 

        :returns: Matplotlib axis object.
    '''
    def plot_iterations_vs_converged(self, per_iteration_convergence:np.ndarray, size = (10, 10), title:str = None):
        import numpy as np
        import matplotlib.pyplot as plt

        plot_main_title_font = {
            'family': 'sans serif',
            'color':  'black',
            'weight': 'bold',
            'size': 18
        }

        _, ax = plt.subplots(figsize = (16, 10))

        # Set title.
        if title != None:
            ax.set_title(label = title, loc = 'center', fontdict = plot_main_title_font, pad = 20)

        ax.grid(b = True, which = 'both', axis = 'both')

        # Plot
        for i in range(0, per_iteration_convergence.shape[0]):
            ax.scatter(x = i, y = np.count_nonzero(per_iteration_convergence[i]), marker = '.', s = 100, alpha = 0.7)

        return ax

    

    '''
        Point plot of iterations versus converged ranks.

        :param stats: Matrix of statistics from algorithm runs.
        :param size: Plot size.
        :param title: Plot title. 

        :returns: Matplotlib axis object.
    '''
    def plot_ranks_convergence(self, per_iteration_convergence:np.ndarray, from_ranking:np.int = 0, to_ranking:np.int = 20, size = (10, 10), title:str = None):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        # Convert convergence matrix to Pandas dataframe for easier plotting.
        per_iteration_convergence_df = pd.DataFrame(per_iteration_convergence)

        plot_main_title_font = {
            'family': 'sans serif',
            'color':  'black',
            'weight': 'bold',
            'size': 18
        }

        _, ax = plt.subplots(figsize = size)

        # Set title.
        if title != None:
            ax.set_title(label = title, loc = 'center', fontdict = plot_main_title_font, pad = 20)

        # Hide labels for y axis.
        ax.yaxis.set_major_locator(plt.NullLocator())

        per_iteration_convergence_df.iloc[:, from_ranking:to_ranking].plot(title = title, colormap = 'tab20b', sharex = True, layout = (10, 2), subplots = True, ax = ax)

        return ax