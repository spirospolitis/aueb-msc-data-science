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
    Implements the Tikhonov regularization part of this homework.
'''
class Tikhonov():

    '''
        Constructor
    '''
    def __init__(self, with_progress_bar:bool = False):
        self.__n = None
        self.__A = None
        self.__B = None
        self.__G = None
        self.__F = None

        self.__with_progress_bar = with_progress_bar

    

    '''
        Returns row / column indices (a, b), based on the vector form of a nxn matrix
        (i.e. as if the the vector was reshaped to a nxn matrix).

        :param n: Dimension nxn matrix that would be formed from the vector, if it was
        resheped to a nxn matrix.
        :param i: Vector index.

        :returns: indices a, b.
    '''
    def __r_c_idx_from_linear_idx(self, n:int, i:int):
        s_a_idx = i // n
        s_b_idx = i - n * s_a_idx
    
        return s_a_idx, s_b_idx



    '''
        Computes the Tikhonov regularized solution to ... .

        :param n: nxn matrix dimensions. 
        :param A: Matrix A.
        :param B: Matrix B.
        :param G: Matrix G.
        :param a: Tikhonov regularization parameter.

        :returns: Self, with estimated F matrix computed.
    '''
    def solve(self, n:int, A:np.ndarray, B:np.ndarray, G:np.ndarray, a:int):
        from tqdm import tqdm

        self.__n = n
        self.__A = A
        self.__B = B
        self.__G = G
        self.__a = a

        # Compute SVD of A, B.
        U_A, S_A, V_A_T = np.linalg.svd(self.__A)
        U_B, S_B, V_B_T = np.linalg.svd(self.__B)

        V_A = V_A_T.T
        V_B = V_B_T.T

        # Compute Kronecker product of S_A, S_B, so that we have access to s_{i} s.
        # s is a (n * n) column vector.
        s = np.kron(S_A, S_B)

        def compute_f(s:np.ndarray, progress_bar = None):
            # f is a (n * n) column vector
            f = np.zeros((self.__n * self.__n, 1), dtype = np.float64)

            # Transform G to a (n * n) column vector g.
            g = self.__G.reshape(self.__n * self.__n, 1)

            # f_hat is a (n * n) column vector
            self.__f_hat = np.zeros((self.__n * self.__n, 1), dtype = np.float64)

            # g_hat is a (n * n) column vector
            self.__g_hat = np.zeros((self.__n * self.__n, 1), dtype = np.float64)

            # For all s_{i} s
            for i in range(0, len(s)):
                if progress_bar is not None:
                    progress_bar.update(1)

                # Retrieve the row, column indices of s.
                s_a_idx, s_b_idx = self.__r_c_idx_from_linear_idx(n = self.__n, i = i)

                u_i = np.kron(U_A[:, s_a_idx], U_B[:, s_b_idx]).reshape(self.__n * self.__n, 1)
                v_i = np.kron(V_A[:, s_a_idx], V_B[:, s_b_idx]).reshape(self.__n * self.__n, 1)
                g_hat_i = u_i.T.dot(g)
                s_i = s[i]

                self.__f_hat[i] = (s_i * g_hat_i) / (s_i**2 + a**2)
                self.__g_hat[i] = g_hat_i

                f = f + self.__f_hat[i] * v_i

            return f

        # Progress bar
        if self.__with_progress_bar == True:
            with tqdm(range(0, len(s)), desc = 'Thikhonov regularization (a = %s)' % ('{:.2e}'.format(self.__a))) as progress_bar: 
                    f = compute_f(s, progress_bar)
        else:
            f = compute_f(s)

        self.__F = f.reshape(self.__n, self.__n)

        return self



    @property
    def n(self):
        return self.__n

    @property
    def A(self):
        return self.__A

    @property
    def B(self):
        return self.__B

    @property
    def G(self):
        return self.__G

    @property
    def F(self):
        return self.__F

    @property
    def a(self):
        return self.__a

    @property
    def f_hat(self):
        return self.__f_hat

    @property
    def g_hat(self):
        return self.__g_hat