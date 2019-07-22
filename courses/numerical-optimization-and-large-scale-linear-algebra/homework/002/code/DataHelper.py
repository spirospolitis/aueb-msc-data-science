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
    Implements data generation functions for the homework.
'''
class DataHelper():
    '''
        Ingests the data file of matrices A, B, G, returns the matrices.

        :param file_path: File path.
        :param n: Matrix size.

        :returns: matrices A, B, G.
    '''
    def ingest_data(self, file_path:str, n:int):
        import pandas as pd

        data_df = pd.read_csv(file_path, sep = ' ', header = None, dtype = np.float64)
        data_df = data_df.drop([4], axis = 1)

        data = data_df.values.flatten()

        # Kronecker product matrices.
        A = data[0:n*n].reshape(n, n)
        B = data[n*n:2*n*n].reshape(n, n)

        # Blurred, noisy image G
        G = data[2*n*n:3*n*n].reshape(n, n)
        
        return A, B, G



    def mp_runner(self, module_name, class_name, parameter_values, *args):
        import multiprocessing as mp
        import importlib

        # Module, class of computation,
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)

        # Method arguments.
        n = args[0]
        A = args[1]
        B = args[2]
        G = args[3]

        # Spawn parallel processes.
        with mp.Pool() as mp_pool:
            mp_pool_executors = [mp_pool.apply_async(class_().solve, args = (n, A, B, G, parameter_value)) for parameter_value in parameter_values]
            mp_pool_results = [mp_pool_executor.get() for mp_pool_executor in mp_pool_executors]

        return mp_pool_results



    # '''
    #     Performs SVD on matrix A, then truncates U, S, V accordingly.

    #     We can find a reduced (k) rank approximation of A by setting 
    #     all but the first k largest singular values equal to zero 
    #     and using only the first k columns of U and V.
        
    #     Truncation is based on the first k s_{i} values of S.

    #     :param A: Matrix on which to perform SVD.
    #     :param p: p first s_{i} to keep (TSVD truncation parameter).
    # '''
    # def truncate_USV(self, U:np.ndarray, S:np.ndarray, V:np.ndarray, p:int):
    #     return U[:, 0:p], np.diag(S[0:p]), V[0:p, :]



    # '''
    #     Computes the estimate of F.
    # '''
    # def F_TSVD(self, G:np.ndarray, U:np.ndarray, S:np.ndarray, V:np.ndarray, alpha:int = 0):
    #     import numpy as np

    #     n = G.shape[0]

    #     g = G.reshape(n * n, 1)

    #     g_hat = np.dot(U.T, g)
    #     f_hat = np.zeros(len(g_hat))

    #     for i in range(len(g_hat)):
    #         f_hat[i] = (S[i][i] * g_hat[i]) / S[i][i]**2 + (alpha**2)

    #     F = np.dot(V.T, f_hat).reshape((n, n))

    #     return F