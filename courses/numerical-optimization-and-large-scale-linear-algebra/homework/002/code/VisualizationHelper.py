'''
    AUEB M.Sc. in Data Science (part-time)
    Course: Numerical Optimization and Large Scale Linear Algebra
    Semester: Spring 2019
    Instructor: P. Vassalos
    Student: S. Politis
    Student ID: p3351814
    Date: 12/06/2019

    Homework 2
'''

import numpy as np

'''
    Implements visualization functions for the homework.
'''
class VisualizationHelper():

    '''
        Point plot.

        :param A: Matrix to plot.
        :param size: Plot size.
        :param title: Plot title. 

        :returns: Matplotlib axis object.
    '''
    def plot_points(self, A:np.ndarray, size = (10, 10), title:str = None):
        import matplotlib.pyplot as plt
        
        plot_main_title_font = {
            'family': 'sans serif',
            'color':  'black',
            'weight': 'bold',
            'size': 18
        }

        fig, ax = plt.subplots(figsize = size)
        
        # Set title.
        if title != None:
            ax.set_title(label = title, loc = 'center', fontdict = plot_main_title_font, pad = 20)
        
        # Axis ticks.
        ax.get_xaxis().set_ticks([0, A.shape[0]])
        ax.get_yaxis().set_ticks([np.min(A), np.max(A)])
        
        ax.get_xaxis().set_major_locator(plt.MultipleLocator(len(A)))
        ax.get_xaxis().set_minor_locator(plt.LinearLocator(numticks = len(A) // 500))
        
        ax.get_yaxis().set_major_locator(plt.MultipleLocator(np.max(A)))
        ax.get_yaxis().set_minor_locator(plt.LinearLocator(numticks = 10))

        ax.grid(b = True, which = 'both', axis = 'both')

        ax.plot(A, color = 'grey', marker = '.', markerfacecolor = 'orange', markersize = 10, alpha = 0.5)

        return ax


    '''
        Plots an image.

        :param A: Matrix to plot.
        :param size: Plot size.
        :param title: Plot title. 

        :returns: Matplotlib axis object.
    '''
    def plot_image(self, A:np.ndarray, size = (10, 10), title:str = None):
        import matplotlib.pyplot as plt
        
        plot_main_title_font = {
            'family': 'sans serif',
            'color':  'black',
            'weight': 'bold',
            'size': 18
        }

        fig, ax = plt.subplots(figsize = size)
        
        # Set title.
        if title != None:
            ax.set_title(label = title, loc = 'center', fontdict = plot_main_title_font, pad = 20)
        
        # Remove axis ticks.
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        
        plt.imshow(A, cmap = 'gray')
        
        return ax