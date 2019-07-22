'''
    AUEB M.Sc. in Data Science (part-time)
    Course: Numerical Optimization and Large Scale Linear Algebra
    Semester: Spring 2019
    Instructor: P. Vassalos
    Student: S. Politis
    Student ID: p3351814
    Date: 18/06/2019

    Homework 3
'''

import numpy as np

class PageRank():

    '''
        Constructor
    '''
    def __init__(self):
        self.__P = None
        self.__alpha = None
        self.__tol = None
        self.__iterations = 0
        self.__x = None
        self.__ranks = None
        self.__per_iteration_convergence = None
        self.__stats = None
        self.__verbose = False



    '''
        Implements the Power Method for the homework.

        Numerically estimates the largest eigenvalues and eigenvectors of a matrix P.

        :param P: Transition matrix P.
        :param alpha: Personalization constant.
        :param tol: Condition of convergence for the power method.

        :returns: PageRank class with populated values.
    '''
    def __solve_power_method(self, P:np.ndarray, alpha:np.float64 = 0.85, max_iterations:int = np.iinfo(np.int32).max, tol:np.float = 1e-8):
        import time
        import numpy as np
        import scipy.sparse

        # Initialize class properties
        self.__P = P
        self.__alpha = alpha
        self.__tol = tol
        self.__iterations = 1

        n = self.__P.shape[0]
        
        # Get the vector a of equation (1), par 5.1.
        # This is the n-vector, containing 0 where there is no dangling node,
        # 1 otherwise.
        a = np.asarray((np.sum(self.__P, axis = 1)[:, 0] == 0).astype(np.int32))
        
        # The n-vector of equation (1), par 5.1.
        v = scipy.sparse.csr_matrix(
            np.ones(shape = (n, 1), dtype = np.float64) / n
        )
        
        # Initial vector for the power iteration.
        x_0 = scipy.sparse.csr_matrix(
            np.ones(shape = (n, 1), dtype = np.float64) / n
        )
        
        # Convergence exit condition.
        not_converged = True

        # Repeat until convergence.
        while not_converged and self.__iterations < max_iterations:
            iteration_start_time = time.time()

            self.__x = (self.__alpha * x_0.T).dot(self.__P) + ((self.__alpha * x_0.T).dot(a) + (1 - self.__alpha))[0, 0] * v.T

            # Convergence condition.
            error = np.linalg.norm((self.__x.T - x_0).toarray(), ord = 2) / np.linalg.norm(self.__x.T.toarray(), ord = 2)
            
            # In order to keep track of convergence we store, per iteration, 
            # # the vector x, sorted in descending order.
            self.__per_iteration_convergence.append((np.argsort(-np.asarray(self.__x.todense())) + 1).reshape(n))

            x_0 = self.__x.T.copy()

            # Exit condition.
            if error <= self.__tol:
                not_converged = False

            # Store iteration statistics (time, error).
            self.__stats.append([self.__iterations, error])

            if self.__verbose == True:
                iteration_time = (time.time() - iteration_start_time) * 1000

                print('Iteration', self.__iterations, '\t\t', 'Time (millis)', str(int(iteration_time)), '\t\t', 'Error', error)

            # Increase iteration counter.
            self.__iterations += 1
        
        # Compute the vector of page ranks.
        # Adding one turns zero-based indexing to page indexing.
        self.__ranks = (np.argsort(-np.asarray(self.__x.todense())) + 1).reshape(n)

        self.__per_iteration_convergence = np.array(self.__per_iteration_convergence)

        return self



    '''
        Implements the solution of the homogenous linear system (par. 5) for the homework.

        :param P: Transition matrix P.
        :param alpha: Personalization constant.
        :param tol: Condition of confergence for the power method.

        :returns: PageRank class with populated values.
    '''
    def __solve_gauss_seidel(self, P:np.ndarray, alpha:np.float64 = 0.85, max_iterations:int = np.iinfo(np.int32).max, tol:np.float = 1e-8):
        import time
        import numpy as np
        import scipy.sparse

        # Initialize class properties
        self.__P = P
        self.__alpha = alpha
        self.__tol = tol
        self.__iterations = 1

        # Convergence exit flag.
        not_converged = True

        n = self.__P.shape[0]

        # Matrix A
        I_aP = (scipy.sparse.eye(self.__P.shape[0], format = 'csc') - alpha * self.__P).T
        
        # Matrix M_G
        I_aP_L = scipy.sparse.tril(I_aP, k = 0, format = 'csc') 
        
        # Matrix R_G
        I_aP_U = - scipy.sparse.triu(I_aP, k = 1, format = 'csc')
        
        v = scipy.sparse.csc_matrix(np.ones((n, 1)) / n)
        
        b = scipy.sparse.csc_matrix(((1 - alpha) * v))
        
        x_0 = scipy.sparse.csc_matrix(np.ones((n, 1)) / n)

        # Repeat until convergence.
        while not_converged and self.__iterations < max_iterations:
            iteration_start_time = time.time()

            self.__x = x_0.copy()
            
            self.__x = scipy.sparse.linalg.spsolve(I_aP_L, I_aP_U.dot(self.__x.reshape(n, 1)) + b.reshape(n, 1))
            
            error = np.linalg.norm(self.__x.reshape(n, 1) - x_0.reshape(n, 1), ord = 1)
            
            # In order to keep track of convergence we store, per iteration, 
            # # the vector x, sorted in descending order.
            self.__per_iteration_convergence.append((np.argsort(-np.asarray(self.__x)) + 1).reshape(n))

            # Compute the error and convergence condition.
            if error <= tol:
                not_converged = False

            x_0 = self.__x

            self.__stats.append([self.__iterations, error])

            if self.__verbose == True:
                iteration_time = (time.time() - iteration_start_time) * 1000

                print('Iteration', self.__iterations, '\t\t', 'Time (millis)', str(int(iteration_time)), '\t\t', 'Error', error)

            self.__iterations += 1

        self.__x = self.__x / np.linalg.norm(self.__x, ord = 1)

        # Compute the vector of page ranks.
        # Adding one turns zero-based indexing to page indexing.
        self.__ranks = np.argsort(-np.asarray(self.__x), axis = 0) + 1

        self.__per_iteration_convergence = np.array(self.__per_iteration_convergence)

        return self


    def solve(self, P:np.ndarray, alpha:np.float64 = 0.85, max_iterations:int = np.iinfo(np.int32).max, tol:np.float = 1e-8, method:str = 'power', verbose = False):
        # Check params.
        if method not in ['power', 'gauss-seidel']:
            raise ValueError("Solve method is either 'power' or 'gauss-seidel'")
        
        # Reset class variables.
        self.__P = None
        self.__alpha = None
        self.__tol = None
        self.__iterations = 0
        self.__x = None
        self.__ranks = None
        self.__per_iteration_convergence = []
        self.__stats = []
        self.__verbose = verbose

        if method == 'power':
            return self.__solve_power_method(P, alpha, max_iterations, tol)

        if method == 'gauss-seidel':
            return self.__solve_gauss_seidel(P, alpha, max_iterations, tol)

        

    '''
        Class properties.
    '''
    @property
    def P(self):
        return self.__P

    @property
    def alpha(self):
        return self.__alpha

    @property
    def tol(self):
        return self.__tol

    @property
    def iterations(self):
        return self.__iterations

    @property
    def x(self):
        return self.__x

    @property
    def ranks(self):
        return self.__ranks

    @property
    def stats(self):
        return np.array(self.__stats)

    @property
    def per_iteration_convergence(self):
        return self.__per_iteration_convergence
