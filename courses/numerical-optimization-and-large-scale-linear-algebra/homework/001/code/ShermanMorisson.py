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

import numpy as np
from tqdm import tqdm

from DataHelper import DataHelper

'''
    Implements the Sherman-Morisson method.
'''
class ShermanMorisson():
    def __init__(self):
        self.__data_helper = DataHelper()
        self.__n = None
        self.__P = None
        self.__L = None
        self.__U = None
        self.__b = None
        self.__u = None
        self.__v = None



    '''
        Sherman-Morisson solution.

        :param n: Problem size.
        :param P: P matrix.
        :param L: L matrix.
        :param U: U matrix.
        :param b: b vector.

        :returns: Solution x vector.
    '''
    def solve(self, n:int, P:np.ndarray, L:np.ndarray, U:np.ndarray, b:np.ndarray):
        self.__n = n
        self.__P = P
        self.__L = L
        self.__U = U
        self.__b = b
        self.__u = self.__data_helper.compute_vector_with_norm_one(n = self.__n, norm = 2)
        self.__v = self.__data_helper.compute_vector_with_norm_one(n = self.__n, norm = 2)

        y = np.linalg.solve(np.dot(self.__L, self.__U), np.dot(self.__P, self.__b))
        z = np.linalg.solve(np.dot(self.__L, self.__U), self.__u)
        a_1 = z * self.__v.T * y
        a_2 = 1.0 + self.__v.T * z

        self.__x = y - (a_1 / a_2)

        return self

    

    '''
        Class properties.
    '''
    @property
    def n(self):
        return self.__n

    @property
    def P(self):
        return self.__P

    @property
    def L(self):
        return self.__L

    @property
    def U(self):
        return self.__U

    @property
    def b(self):
        return self.__b

    @property
    def u(self):
        return self.__u

    @property
    def v(self):
        return self.__v

    @property
    def x(self):
        return self.__x
