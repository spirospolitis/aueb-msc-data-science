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

'''
    Implements the Gaussian elimination solver with no, partial or full pivoting.
'''
class GaussianElimination():
    def __init__(self, with_progress_bar = False):
        self.__A = None
        self.__b = None
        self.__n = None
        self.__P = None
        self.__Q = None
        self.__L = None
        self.__U = None
        self.__x = None

        self.__with_progress_bar = with_progress_bar



    '''
        Computes permutations for partial and full pivoting.
        
        :param pivoting: Pivoting strategy to use:
            No pivoting = None
            Partial pivoting = 'partial'
            Full pivoting = 'full'
        
        :returns: 2-element vector of indices that should be exchanged.
    '''
    def __pivoting(self, pivoting:str = None):
        if pivoting == None:
            return

        # Retrieve the diagonal indices of U ({{0, 0}, {1, 1}, ... {u_ij, u_ij}}, i = j)
        U_diag_indices = np.diag_indices(self.__U.shape[0])

        def __do_pivoting(pivoting:str, progress_bar = None):
            # Retrieve the diagonal indices of U ({{0, 0}, {1, 1}, ... {u_ij, u_ij}}, i = j)
            U_diag_indices = np.diag_indices(self.__U.shape[0])

            # For every column except the last in U...
            for j in U_diag_indices[1][:-1]:
                
                if progress_bar is not None:
                    progress_bar.update(1)

                # Setting r (row) = c (column) = j just to preserve semantics.
                r = j
                c = j
            
                # Partial pivoting
                if pivoting == 'partial':
                    # Create a 2-element array to store the row exchange indices.
                    # P = np.zeros(2, dtype = np.int32)

                    # Temporary matrix with the absolute values of U.
                    U_abs = np.abs(self.__U)

                    # Retrieve the row index of the column we are examining in the current step,
                    # such that it contains the number with the maximum absolute value.
                    # We add the current column to the index variable so as to consider only values
                    # below the diagonal (including the diagonal element).
                    argmax_idx = r + np.argmax(U_abs[r:, c])
                    
                    # Apply row pivoting on P
                    self.__P[[r, argmax_idx]] = self.__P[[argmax_idx, r]]

                    # Apply row pivoting on U
                    self.__U[[r, argmax_idx]] = self.__U[[argmax_idx, r]]

                    # Apply row pivoting on L
                    self.__L[[r, argmax_idx], :c] = self.__L[[argmax_idx, r], :c]

                # Full pivoting
                if pivoting == 'full':
                    # Subset of matrix to examine for max absolute value.
                    U_abs = np.abs(self.__U)[r:, c:]

                    # Find the element a_ij with the absolute max value in A,
                    # such that i, j > diag(i, j).

                    # Find max absolute value on matrix subset.
                    argmax_idx = np.unravel_index(np.argmax(U_abs), U_abs.shape)
                    
                    # Re-align the indices to the original matrix
                    argmax_idx_norm = tuple([sum(x) for x in zip(argmax_idx, (r, c))])
                    
                    # Apply row pivoting on P
                    self.__P[[r, argmax_idx_norm[0]]] = self.__P[[argmax_idx_norm[0], r]]

                    # Apply column pivoting on Q
                    # self.__Q.T[[c, argmax_idx_norm[1]]] = self.__Q.T[[argmax_idx_norm[1], c]]
                    self.__Q[:, [c, argmax_idx_norm[1]]] = self.__Q[:, [argmax_idx_norm[1], c]]

                    # Apply row and column pivoting on U
                    self.__U[[r, argmax_idx_norm[0]]] = self.__U[[argmax_idx_norm[0], r]]
                    self.__U[:, [c, argmax_idx_norm[1]]] = self.__U[:, [argmax_idx_norm[1], c]]

                    # Apply row pivoting on L
                    self.__L[[r, argmax_idx_norm[0]], :c] = self.__L[[argmax_idx_norm[0], r], :c]

        if self.__with_progress_bar == True:
            with tqdm(range(0, len(U_diag_indices[1][:-1])), desc = 'Pivoting (%s, n = %d)' % (pivoting, self.__n)) as progress_bar:
                __do_pivoting(pivoting, progress_bar)
        else:
            __do_pivoting(pivoting)

                

    '''
        Implements the Gaussian elimination algorithm.
        
        :returns: 
    '''
    def __elimination(self):
        # Retrieve the diagonal indices of U ({{0, 0}, {1, 1}, ... {u_ij, u_ij}}, i = j)
        U_diag_indices = np.diag_indices(self.__U.shape[0])

        def __do_elimination(progress_bar = None):
            U_diag_indices = np.diag_indices(self.__U.shape[0])
            
            # For every column except the last in U...
            for j in U_diag_indices[1][:-1]:
                
                if progress_bar is not None:
                    progress_bar.update(1)

                # For every i - 1 row in U...
                for i in reversed(U_diag_indices[0]):

                    diag_i = diag_j = j
                        
                    # We are operating on elements below the diagonal
                    if i > j:
                            
                        # If the element is 0, there is no need to proceed with elimination
                        if self.__U[i, j] == 0:
                            
                            progress_bar.update(1)

                            continue
                        
                        # Compute the quantity that will eliminate u_ij
                        u = -self.__U[i, j] / self.__U[diag_i, diag_j]
                        
                        # Act on U
                        self.__U[i, :] = (u * self.__U[diag_i, :]) + self.__U[i, :]
                        
                        # Store the u quantity in L
                        self.__L[i, j] = -u

        # Progress bar
        if self.__with_progress_bar:
            with tqdm(range(0, len(U_diag_indices[1][:-1])), desc = 'Elimination (n = %d)' % self.__n) as progress_bar:
                __do_elimination(progress_bar)
        else:
            __do_elimination()



    '''
        Solves A.x = b.

        :param A: nxn matrix of coefficients of the system of equations.
        :param b: n-vector of system solutions.
        :param pivoting: Pivoting strategy to use:
            No pivoting = None
            Partial pivoting = 'partial'
            Full pivoting = 'full'

        :returns: Fitted version of the GaussianElimination class.
    '''
    def solve(self, A:np.ndarray, b:np.ndarray, pivoting:str = None):
        # Input checks.
        if A.shape[0] != A.shape[1]:
            raise ValueError('A should be a nxn matrix.')
        
        if A.shape[0] != b.shape[0]:
            raise ValueError('A, b have incompatible dimensions.')

        # Input variables
        self.__A = A
        self.__b = b

        self.__n = self.__A.shape[0]

        self.__P = np.eye(self.__A.shape[0], dtype = np.int32)

        self.__Q = np.eye(self.__A.shape[0], dtype = np.int32)

        # Initialize L as the I_{n} identity matrix.
        self.__L = np.eye(self.__A.shape[0], dtype = np.float64)
        
        # Initialize U as a copy of A.
        self.__U = self.__A.copy()

        self.__x = None
        
        # Perform pivoting.
        self.__pivoting(pivoting)
        
        # Perform elimination.
        self.__elimination()

        ###
        # NEW SOLUTION
        ##
        # y = Pb
        # Lc = y
        # Ux = c
        if pivoting == 'partial':
            y = np.dot(self.__P, self.__b)
            c = np.linalg.solve(self.__L, y)
            self.__x = np.linalg.solve(self.__U, c)

        # y = Pb
        # Lc = y
        # Uz = c
        # x = Qz
        if pivoting == 'full':
            y = np.dot(self.__P, self.__b)
            c = np.linalg.solve(self.__L, y)
            z = np.linalg.solve(self.__U, c)
            self.__x = np.linalg.solve(self.__Q.T, z)

        ###
        # NEW SOLUTION
        ##

        ###
        # LAST KNOWN SOLUTION
        ##
        # Solve the system Ly = Pb
        # y = np.linalg.solve(self.__L, np.dot(self.__P, self.__b))

        # # Solve the system Uz = y
        # z = np.linalg.solve(self.__U, y)

        # # Solve the system Q.Tx = z
        # self.__x = np.linalg.solve(self.__Q.T, z)
        ###
        # LAST KNOWN SOLUTION
        ##

        return self



    '''
        Class properties.
    '''
    @property
    def A(self):
        return self.__A

    @property
    def b(self):
        return self.__b

    @property
    def n(self):
        return self.__n

    @property
    def P(self):
        return self.__P

    @property
    def Q(self):
        return self.__Q

    @property
    def L(self):
        return self.__L

    @property
    def U(self):
        return self.__U

    @property
    def x(self):
        return self.__x

    @property
    def PAQ(self):
        return np.dot(self.P, np.dot(self.__A, self.Q))

    @property
    def LU(self):
        return np.dot(self.__L, self.__U)