'''
    AUEB M.Sc. in Data Science (part-time)
    Course: Numerical Optimization and Large Scale Linear Algebra
    Semester: Spring 2019
    Instructor: P. Vassalos
    Student: S. Politis
    Student ID: 
    Date: 15/05/2019

    Homework 1
'''

import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm

'''
    Implements visualization functions for the homework.
'''
class VisualizationHelper():

    '''
        Plots problem size vs errors.


    '''
    def plot_n_vs_e(self, iteration_results:[], y:str, label:str, dot_scaling:float = 1, normalize_axes:bool = False):
        n = np.array([iteration.get('n') for iteration in iteration_results[0]])
        y_param_1 = np.array([iteration.get(y) for iteration in iteration_results[0]])
        y_param_2 = np.array([iteration.get(y) for iteration in iteration_results[1]])

        if normalize_axes == True:
            n = n / np.linalg.norm(n)
            y_param_1 = y_param_1 / np.linalg.norm(y_param_1)
            y_param_2 = y_param_2 / np.linalg.norm(y_param_2)

        plot_main_title_font = {
            'family': 'sans serif',
            'color':  'black',
            'weight': 'bold',
            'size': 18
        }

        plot_axis_title_font = {
            'family': 'sans serif',
            'color':  'black',
            'weight': 'bold',
            'size': 14
        }

        # Plot.
        fig, ax = plt.subplots(figsize = (18, 10))

        ax.scatter(x = n, y = y_param_1, s = 10 * dot_scaling, alpha = 0.5, cmap = 'viridis', label = 'partial pivoting')
        ax.scatter(x = n, y = y_param_2, s = 10 * dot_scaling, alpha = 0.5, cmap = 'viridis', label = 'full pivoting')

        # Set graph title.
        ax.set_title(label = label, loc = 'center', fontdict = plot_main_title_font, pad = 20)

        # Set axis proper labels.
        ax.set_xlabel(xlabel = 'Μέγεθος προβλήματος', fontdict = plot_axis_title_font, labelpad = 20)

        # Set x-axis ticks and corresponding labels to be the index of our series object.
        ax.set_xticks(ticks = [iteration.get('n') for iteration in iteration_results[0]])
        ax.set_xticklabels(labels = [iteration.get('n') for iteration in iteration_results[0]])
        
        # Set x-axis scale.
        ax.set_xscale('linear')

        # Show grid.
        ax.grid(True)

        # Show legend.
        ax.legend()

        return fig, ax




    '''
        Plots problem size vs errors.


    '''
    def plot_n_vs_mean_e(self, iteration_results:[], y:str, label:str, dot_scaling:float = 1, normalize_axes:bool = False):
        n = np.unique(np.array([iteration.get('n') for iteration in iteration_results[0]]))

        # Compute mean error per problem size, methid type
        b = []
        for iteration in iteration_results[0]:
            a = np.zeros((2))
            a[0] = iteration.get('n')
            a[1] = iteration.get(y)
            b.append(a)

        y_param_1 = pd.DataFrame(np.array(b)).groupby(by = 0).mean().values

        b = []
        for iteration in iteration_results[1]:
            a = np.zeros((2))
            a[0] = iteration.get('n')
            a[1] = iteration.get(y)
            b.append(a)

        y_param_2 = pd.DataFrame(np.array(b)).groupby(by = 0).mean().values


        if normalize_axes == True:
            n = n / np.linalg.norm(n)
            y_param_1 = y_param_1 / np.linalg.norm(y_param_1)
            y_param_2 = y_param_2 / np.linalg.norm(y_param_2)

        plot_main_title_font = {
            'family': 'sans serif',
            'color':  'black',
            'weight': 'bold',
            'size': 18
        }

        plot_axis_title_font = {
            'family': 'sans serif',
            'color':  'black',
            'weight': 'bold',
            'size': 14
        }

        # Plot.
        fig, ax = plt.subplots(figsize = (18, 10))

        ax.scatter(x = n, y = y_param_1, s = 10 * dot_scaling, alpha = 0.5, cmap = 'viridis', label = 'partial pivoting mean error')
        ax.scatter(x = n, y = y_param_2, s = 10 * dot_scaling, alpha = 0.5, cmap = 'viridis', label = 'full pivoting mean error')

        # Set graph title.
        ax.set_title(label = label, loc = 'center', fontdict = plot_main_title_font, pad = 20)

        # Set axis proper labels.
        ax.set_xlabel(xlabel = 'Μέγεθος προβλήματος', fontdict = plot_axis_title_font, labelpad = 20)

        # Set x-axis ticks and corresponding labels to be the index of our series object.
        ax.set_xticks(ticks = [iteration.get('n') for iteration in iteration_results[0]])
        ax.set_xticklabels(labels = [iteration.get('n') for iteration in iteration_results[0]])
        
        # Set x-axis scale.
        ax.set_xscale('linear')

        # Show grid.
        ax.grid(True)

        # Show legend.
        ax.legend()

        return fig, ax





    '''
        :param iteration_results: Array of parameters of different problem executions results.
        :param x: x-axis variable.
        :param y: y-axis variable.
        :param label: Plot label.

        :returns: Matplotlib figure
    '''
    def plot_k_vs_e(self, iteration_results:[], x:str, y:str, x_label:str, y_label:str, label:str, with_lines:bool = False, interpolation_factor:float = None, dot_scaling:float = 1, normalize_axes:bool = False):
        from operator import itemgetter

        if x == 'K_A':
            # Sort array of dicts on key x_param (K_A in this case).
            # The reason for doing this is to sort on condition number of A.
            # iteration_results = sorted(iteration_results, key = itemgetter(x)) 

            x_param = [iteration_result.get(x) for iteration_result in iteration_results]

            if y == 'e_c_inf_norm':
                y_param = [iteration_result.get(y) for iteration_result in iteration_results]

            if y == 'e_r_inf_norm':
                y_param = [iteration_result.get(y) for iteration_result in iteration_results]

        plot_main_title_font = {
            'family': 'sans serif',
            'color':  'black',
            'weight': 'bold',
            'size': 18
        }

        plot_axis_title_font = {
            'family': 'sans serif',
            'color':  'black',
            'weight': 'bold',
            'size': 14
        }

        # Plot.
        fig, ax = plt.subplots(figsize = (14, 6))
        
        # Legend label is problem size
        labels = [iteration_result.get('n') for iteration_result in iteration_results]

        # Replace NaN
        x_param = np.array([i if not np.isnan(i) else 0 for i in x_param])
        y_param = np.array([i if not np.isnan(i) else 0 for i in y_param])

        # Replace Inf
        x_param = np.array([i if i != np.inf else 1e+5 for i in x_param])
        y_param = np.array([i if i != np.inf else 1e+5 for i in y_param])

        if normalize_axes == True:
            x_param = x_param / np.linalg.norm(x_param)
            y_param = y_param / np.linalg.norm(y_param)
            s = (x_param / np.linalg.norm(x_param)) * dot_scaling
        else:
            s = (x_param / np.linalg.norm(x_param)) * dot_scaling

        if interpolation_factor != None:
            # Interpolate based on data points.
            interpolation_function = interp1d(x_param / np.linalg.norm(x_param), y_param / np.linalg.norm(y_param), axis = 0, fill_value = 'extrapolate', kind = 'cubic')
            interpolated_data = np.linspace(0, np.power(2, math.log(x_param[len(x_param) - 1], 2) + interpolation_factor), num = 1000, endpoint = True)

            ax.scatter(x = x_param, y = y_param, s = s, c = x_param, alpha = 0.5, label = labels, cmap = 'viridis')

            if with_lines == True:
                ax.plot(interpolated_data, interpolation_function(interpolated_data), '-', c = 'orange')
        else:
            ax.scatter(x = x_param, y = y_param, s = s, c = x_param, alpha = 0.5, label = labels, cmap = 'viridis')

            if with_lines == True:
                ax.plot(x_param / np.linalg.norm(x_param), y_param / np.linalg.norm(y_param), '-', c = 'orange')

        # Set graph title.
        ax.set_title(label = label, loc = 'center', fontdict = plot_main_title_font, pad = 20)

        # Set axis proper labels.
        ax.set_xlabel(xlabel = x_label, fontdict = plot_axis_title_font, labelpad = 20)
        ax.set_ylabel(ylabel = y_label, fontdict = plot_axis_title_font, labelpad = 20)

        # Set x-axis ticks and corresponding labels to be the index of our series object.
        ax.set_xticks(ticks = x_param)
        ax.set_xticklabels(labels = x_param)
        
        plt.xticks(rotation = 90)
        
        # Set x-axis scale.
        ax.set_xscale('linear')

        # Show grid.
        ax.grid(True)

        # Show legend.
        # ax.legend()

        return fig, ax


    '''
        :param iteration_results: Array of parameters of different problem executions results.
        :param x: x-axis variable.
        :param y: y-axis variable.
        :param label: Plot label.

        :returns: Matplotlib figure
    '''
    def plot_n_vs_time(self, iteration_results:[[]], x:str, y:str, x_label:str, y_label:str, label:str, with_lines:bool = False, interpolation_factor:int = None, dot_scaling = 1):
        from operator import itemgetter
        
        plot_main_title_font = {
            'family': 'sans serif',
            'color':  'black',
            'weight': 'bold',
            'size': 18
        }

        plot_axis_title_font = {
            'family': 'sans serif',
            'color':  'black',
            'weight': 'bold',
            'size': 14
        }
        
        # Plot.
        fig, ax = plt.subplots(figsize = (14, 6))

        if len(iteration_results) == 1:
            
            label = iteration_results[0][0]['method']

            if label == 'gauss':
                label = label + ' ' + '(' + iteration_results[0][0]['pivoting'] + ' ' + 'pivoting' + ')'

            if x == 'n':
                x_param_1 = [iteration_result.get(x) for iteration_result in iteration_results[0]]

                if y == 'process_time':
                    y_param_1 = [iteration_result.get(y) for iteration_result in iteration_results[0]]
                
            if interpolation_factor != None:
                # Interpolate based on data points.
                interpolation_function_1 = interp1d(x_param_1, y_param_1, axis = 0, fill_value = 'extrapolate', kind = 'cubic')
                interpolated_data_1 = np.linspace(0, np.power(2, math.log(x_param_1[len(x_param_1) - 1], 2) + interpolation_factor), num = 1000, endpoint = True)
                ax.scatter(x = x_param_1, y = y_param_1, s = (x_param_1 / np.linalg.norm(x_param_1)) * dot_scaling, c = x_param_1, alpha = 0.5, cmap = 'viridis')

                if with_lines == True:
                    ax.plot(interpolated_data_1, interpolation_function_1(interpolated_data_1), '-', c = 'orange', alpha = 0.5, label = label)
            else:
                ax.scatter(x = x_param_1, y = y_param_1, s = (x_param_1 / np.linalg.norm(x_param_1)) * dot_scaling, c = x_param_1, alpha = 0.5, cmap = 'viridis')

                if with_lines == True:
                    ax.plot(x_param_1, y_param_1, '-', c = 'orange', alpha = 0.5, label = label)

        if len(iteration_results) == 2:

            if x == 'n':
                x_param_1 = [iteration_result.get(x) for iteration_result in iteration_results[0]]
                x_param_2 = [iteration_result.get(x) for iteration_result in iteration_results[1]]

                if y == 'process_time':
                    y_param_1 = [iteration_result.get(y) for iteration_result in iteration_results[0]]
                    y_param_2 = [iteration_result.get(y) for iteration_result in iteration_results[1]]

            if interpolation_factor != None:
                # Interpolate based on data points.
                interpolation_function_1 = interp1d(x_param_1, y_param_1, axis = 0, fill_value = 'extrapolate', kind = 'cubic')
                interpolated_data_1 = np.linspace(0, np.power(2, math.log(x_param_1[len(x_param_1) - 1], 2) + interpolation_factor), num = 1000, endpoint = True)
                ax.scatter(x = x_param_1, y = y_param_1, s = (x_param_1 / np.linalg.norm(x_param_1)) * dot_scaling, c = x_param_1, alpha = 0.5, cmap = 'viridis')

                interpolation_function_2 = interp1d(x_param_2, y_param_2, axis = 0, fill_value = 'extrapolate', kind = 'cubic')
                interpolated_data_2 = np.linspace(0, np.power(2, math.log(x_param_2[len(x_param_2) - 1], 2) + interpolation_factor), num = 1000, endpoint = True)
                ax.scatter(x = x_param_2, y = y_param_2, s = (x_param_2 / np.linalg.norm(x_param_2)) * dot_scaling, c = x_param_2, alpha = 0.5, cmap = 'viridis')

                if with_lines == True:
                    ax.plot(interpolated_data_1, interpolation_function_1(interpolated_data_1), '-', c = 'orange', label = iteration_results[0][0]['method'] + ' ' + '(' + iteration_results[0][0]['pivoting'] + ' ' + 'pivoting' + ')')
                    ax.plot(interpolated_data_2, interpolation_function_2(interpolated_data_2), '-', c = 'blue', label = iteration_results[0][0]['method'] + ' ' + '(' + iteration_results[1][0]['pivoting'] + ' ' + 'pivoting' + ')')
            else:
                ax.scatter(x = x_param_1, y = y_param_1, s = (x_param_1 / np.linalg.norm(x_param_1)) * dot_scaling, c = x_param_1, alpha = 0.5, cmap = 'viridis')
                ax.scatter(x = x_param_2, y = y_param_2, s = (x_param_2 / np.linalg.norm(x_param_2)) * dot_scaling, c = x_param_2, alpha = 0.5, cmap = 'viridis')

                if with_lines == True:
                    ax.plot(x_param_1, y_param_1, '-', c = 'orange', alpha = 0.5, label = iteration_results[0][0]['method'] + ' ' + '(' + iteration_results[0][0]['pivoting'] + ' ' + 'pivoting' + ')')
                    ax.plot(x_param_2, y_param_2, '-', c = 'blue', alpha = 0.5, label = iteration_results[1][0]['method'] + ' ' + '(' + iteration_results[1][0]['pivoting'] + ' ' + 'pivoting' + ')')

        if len(iteration_results) == 3:
            if x == 'n':
                x_param_1 = [iteration_result.get(x) for iteration_result in iteration_results[0]]
                x_param_2 = [iteration_result.get(x) for iteration_result in iteration_results[1]]
                x_param_3 = [iteration_result.get(x) for iteration_result in iteration_results[2]]

                if y == 'process_time':
                    y_param_1 = [iteration_result.get(y) for iteration_result in iteration_results[0]]
                    y_param_2 = [iteration_result.get(y) for iteration_result in iteration_results[1]]
                    y_param_3 = [iteration_result.get(y) for iteration_result in iteration_results[2]]

            if interpolation_factor != None:
                # Interpolate based on data points.
                interpolation_function_1 = interp1d(x_param_1, y_param_1, axis = 0, fill_value = 'extrapolate', kind = 'cubic')
                interpolated_data_1 = np.linspace(0, np.power(2, math.log(x_param_1[len(x_param_1) - 1], 2) + interpolation_factor), num = 1000, endpoint = True)
                ax.scatter(x = x_param_1, y = y_param_1, s = (x_param_1 / np.linalg.norm(x_param_1)) * dot_scaling, c = x_param_1, alpha = 0.5, cmap = 'viridis')

                interpolation_function_2 = interp1d(x_param_2, y_param_2, axis = 0, fill_value = 'extrapolate', kind = 'cubic')
                interpolated_data_2 = np.linspace(0, np.power(2, math.log(x_param_2[len(x_param_2) - 1], 2) + interpolation_factor), num = 1000, endpoint = True)
                ax.scatter(x = x_param_2, y = y_param_2, s = (x_param_2 / np.linalg.norm(x_param_2)) * dot_scaling, c = x_param_2, alpha = 0.5, cmap = 'viridis')

                interpolation_function_3 = interp1d(x_param_3, y_param_3, axis = 0, fill_value = 'extrapolate', kind = 'cubic')
                interpolated_data_3 = np.linspace(0, np.power(2, math.log(x_param_3[len(x_param_3) - 1], 2) + interpolation_factor), num = 1000, endpoint = True)
                ax.scatter(x = x_param_3, y = y_param_3, s = (x_param_3 / np.linalg.norm(x_param_3)) * dot_scaling, c = x_param_2, alpha = 0.5, cmap = 'viridis')

                if with_lines == True:
                    ax.plot(interpolated_data_1, interpolation_function_1(interpolated_data_1), '-', c = 'orange', alpha = 0.5, label = iteration_results[0][0]['method'] + ' ' + '(' + iteration_results[0][0]['pivoting'] + ' ' + 'pivoting' + ')')
                    ax.plot(interpolated_data_2, interpolation_function_2(interpolated_data_2), '-', c = 'blue', alpha = 0.5, label = iteration_results[1][0]['method'] + ' ' + '(' + iteration_results[1][0]['pivoting'] + ' ' + 'pivoting' + ')')
                    ax.plot(interpolated_data_3, interpolation_function_3(interpolated_data_3), '-', c = 'green', alpha = 0.5, label = iteration_results[2][0]['method'])
            else:
                ax.scatter(x = x_param_1, y = y_param_1, s = (x_param_1 / np.linalg.norm(x_param_1)) * dot_scaling, c = x_param_1, alpha = 0.5, cmap = 'viridis')
                ax.scatter(x = x_param_2, y = y_param_2, s = (x_param_2 / np.linalg.norm(x_param_2)) * dot_scaling, c = x_param_2, alpha = 0.5, cmap = 'viridis')
                ax.scatter(x = x_param_3, y = y_param_3, s = (x_param_3 / np.linalg.norm(x_param_3)) * dot_scaling, c = x_param_2, alpha = 0.5, cmap = 'viridis')

                if with_lines == True:
                    ax.plot(x_param_1, y_param_1, '-', c = 'orange', alpha = 0.5, label = iteration_results[0][0]['method'] + ' ' + '(' + iteration_results[0][0]['pivoting'] + ' ' + 'pivoting' + ')')
                    ax.plot(x_param_2, y_param_2, '-', c = 'blue', alpha = 0.5, label = iteration_results[1][0]['method'] + ' ' + '(' + iteration_results[1][0]['pivoting'] + ' ' + 'pivoting' + ')')
                    ax.plot(x_param_3, y_param_3, '-', c = 'green', alpha = 0.5, label = iteration_results[2][0]['method'])

        # Set graph title.
        ax.set_title(label = label, loc = 'center', fontdict = plot_main_title_font, pad = 20)

        # Set axis proper labels.
        ax.set_xlabel(xlabel = x_label, fontdict = plot_axis_title_font, labelpad = 20)
        ax.set_ylabel(ylabel = y_label, fontdict = plot_axis_title_font, labelpad = 20)

        # Set x-axis ticks and corresponding labels to be the index of our series object.
        ax.set_xticks(ticks = x_param_1)
        ax.set_xticklabels(labels = x_param_1)
        
        plt.xticks(rotation=90)
        
        # Set x-axis scale.
        ax.set_xscale('linear')

        # Show grid.
        ax.grid(True)

        # Show legend.
        ax.legend()

        # Legend
        #ax.legend(legend_params)
        #ax.legend(tuple(x_param), tuple(legend_params))

        # colors = ['black', 'red', 'green']
        # icons = [Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in colors]
        # labels = ['black data', 'red data', 'green data']
        # fig.legend(x_param, x_param)

        return fig, ax