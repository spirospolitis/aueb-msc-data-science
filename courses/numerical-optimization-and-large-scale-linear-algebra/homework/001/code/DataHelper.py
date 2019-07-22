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
    Implements data generation functions for the homework.
'''
class DataHelper():

    '''
        Returns a system Ax = b, such that A, x are random and b is the product Ax.

        :param n: problem size.

        :returns: matrix A, vector X and vector b, such that A.x = b.
    '''
    def compute_A_x_b(self, n:int):
        import numpy as np

        # Compute a random matrix A.
        A = np.random.random(size = (n, n))

        # Compute a random vector x.
        x = np.random.random(size = n)

        b = np.dot(A, x)

        # Return A, x as well as the dot product b = A.x.
        return A, x, b



    '''
        Returns a system Ax = b, such that A is known, x is random and b is the product Ax.

        :param A: matrix A.

        :returns: matrix A, vector X and vector b, such that A.x = b.
    '''
    def compute_x_b(self, A:np.ndarray):
        import numpy as np

        n = A.shape[0]

        # Compute a random vector x.
        x = np.random.random(size = n)

        b = np.dot(A, x)

        # Return A, x as well as the dot product b = A.x.
        return A, x, b



    '''
        Computes the system of equations, as required by part 2 of the exersise.
        
        :parameter problem_sizes: problem sizes (n = 64, 128, ...)

        :returns: Array of dictionary objects with n, A, x, b components.
    '''
    def compute_systems_part_2(self, problem_sizes:np.ndarray):
        import numpy as np

        # Holding dicts of n, A, x, b 's.
        systems = []

        for i in range(0, len(problem_sizes)):
            A, x, b = self.compute_A_x_b(problem_sizes[i])

            systems.append({
                'n': problem_sizes[i],
                'A': A,
                'x': x,
                'b': b
            })
        
        return systems



    '''
        Computes the required matrix A, as spcified in part 3 of the exercise.
        
        :parameter n: size of the the nxn matrix.
        :returns: Numpy nxn matrix.
    '''
    def compute_A_part_3(self, n:int):
        import numpy as np

        # Create a lower triangular matrix, based on the identity matrix.
        A = np.tril(np.eye(n, dtype = np.float64))
        
        # Set all elements below the diagonal to -1.
        A[np.tril_indices(n, k = -1)] = -1.0
        
        # Set last column elements to 1.
        A[:, -1] = 1.0
        
        return A



    '''
        Computes the systems of equations, as required by part 3 of the exersise.
        
        :parameter problem_sizes: problem sizes (n = 64, 128, ...)

        :returns: Array of dictionary objects with n, A, x, b components.
    '''
    def compute_systems_part_3(self, problem_sizes:np.ndarray):
        import numpy as np

        # Holding dicts of n, A, x, b 's.
        systems = []

        for i in range(0, len(problem_sizes)):
            A, x, b = self.compute_x_b(self.compute_A_part_3(problem_sizes[i]))

            systems.append({
                'n': problem_sizes[i],
                'A': A,
                'x': x,
                'b': b
            })
        
        return systems



    '''
        Computes a vector of specified norm length.

        :param n: Vector size.
        :param norm: The norm to comply to (default is L2).

        :returns: Vector v.
    '''
    def compute_vector_with_norm_one(self, n:int, norm:int = 2):
        import numpy as np
        
        if n <= 0:
            raise ValueError('Size n cannot be less or equal to 0.')

        v = np.random.uniform(0, 1, n)
        v = v / np.linalg.norm(v, norm)
        
        return v



    '''
        Function designed so as to execute Gaussian iterations of different problem sizes and compute the following metrics:

        method                  Gaussian eliminatiopn method (partial or full pivoting)
        process_time            Execution time
        n                       Problem size (e.g. 64, 128 etc.)
        A                       A matrix
        x                       x vector
        b                       b vector
        P                       P matrix
        L                       L matrix
        U                       U matrix
        e_c                     Computation error
        e_c_inf_norm            Computation error inf norm
        e_r                     Reconstruction error
        e_r_inf_norm            Reconstruction error inf norm
        K_A                     Matrix A condition number

        :param systems: Systems of equations upon which to perform iterations.
        :param pivoting: Pivoting method

        :returns: Array of execution results.
    '''
    def compute_gauss_iterations(self, systems:[], pivoting:str = 'partial', with_progress_bar = False):
        import time
        import numpy as np

        from GaussianElimination import GaussianElimination

        iteration_results = []

        for i in range(0, len(systems)):
            
            gaussian_elimination = GaussianElimination(with_progress_bar = with_progress_bar)

            iteration_result = dict()
            
            n = systems[i]['n'] 
            A = systems[i]['A']
            x = systems[i]['x']
            b = systems[i]['b']
            
            # Perform Gaussian elimination while tracking execution time in milliseconds.
            process_start_time = time.time()
            gaussian_elimination = gaussian_elimination.solve(A, b, pivoting = pivoting)
            total_process_time = (time.time() - process_start_time) * 1000

            # Computation method            
            iteration_result['method'] = 'gauss'

            # Pivoting method            
            iteration_result['pivoting'] = pivoting

            # Total process time
            iteration_result['process_time'] = total_process_time

            # Problem size
            iteration_result['n'] = n

            # Matrix A
            iteration_result['A'] = A

            # Vector x
            iteration_result['x'] = x

            # Vector b
            iteration_result['b'] = b

            # Matrix P
            iteration_result['P'] = gaussian_elimination.P

            # Matrix Q
            iteration_result['Q'] = gaussian_elimination.Q

            # Matrix L
            iteration_result['L'] = gaussian_elimination.L

            # Matrix U
            iteration_result['U'] = gaussian_elimination.U

            # Estimated x
            iteration_result['x_hat'] = gaussian_elimination.x

            # Estimated b
            iteration_result['b_hat'] = np.dot(np.dot(iteration_result['L'], iteration_result['U']), iteration_result['x_hat'])

            # Matrix A condition number
            iteration_result['K_A'] = np.linalg.cond(iteration_result['A'], np.inf)

            # Computation error
            # x - x_hat
            iteration_result['e_c'] = iteration_result['x'] - iteration_result['x_hat']

            # Computation error inf norm
            iteration_result['e_c_inf_norm'] = np.linalg.norm(iteration_result['e_c'], np.inf)

            # Correction error
            # b - A * x_hat 
            # b_hat = A.dot(x_hat.T)
            iteration_result['e_r'] = iteration_result['b'] - np.dot(iteration_result['A'], iteration_result['x_hat'].T)

            # Correction error inf norm
            iteration_result['e_r_inf_norm'] = np.linalg.norm(iteration_result['e_r'], np.inf)

            iteration_results.append(iteration_result)
        
        return iteration_results



    '''
        Function designed so as to execute Sherman-Morisson iterations of different problem sizes and compute the following metrics:

        n                   Problem size (e.g. 64, 128 etc.)
        process_time        Execution time
        P                   Sherman-Morisson P matrix
        L                   Sherman-Morisson U matrix
        U                   Sherman-Morisson U matrix
        b                   Sherman-Morisson b vector
        u                   Sherman-Morisson u vector
        v                   Sherman-Morisson v vector
        x                   Sherman-Morisson x (solution) vector

        Although P, L, U, b, u, v, x are not essential for the purpose of computing asymptotic time,
        they are included for completeness and, perhaps, verification purposes.

        :param pp_iteration_results: iteration results produced by Gaussian iterations of previous questions.

        :returns: Array of execution results.
    '''
    def compute_sherman_morisson_iterations(self, pp_iteration_results:[]):
        import time
        import numpy as np

        from ShermanMorisson import ShermanMorisson

        iteration_results = []

        for i in range(0, len(pp_iteration_results)):
            
            sherman_morisson = ShermanMorisson()

            iteration_result = dict()
            
            n = pp_iteration_results[i]['n'] 
            P = pp_iteration_results[i]['P']
            L = pp_iteration_results[i]['L']
            U = pp_iteration_results[i]['U']
            b = pp_iteration_results[i]['b']

            # Perform Sherman-Morisson runs while tracking execution time in milliseconds.
            process_start_time = time.time()
            gaussian_elimination = sherman_morisson.solve(n, P, L, U, b)
            total_process_time = (time.time() - process_start_time) * 1000

            # Computation method            
            iteration_result['method'] = 'sherman-morisson'

            # Problem size
            iteration_result['n'] = n

            # Total process time
            iteration_result['process_time'] = total_process_time

            # Sherman-Morisson P matrix
            iteration_result['P'] = sherman_morisson.P

            # Sherman-Morisson L matrix
            iteration_result['L'] = sherman_morisson.L

            # Sherman-Morisson U matrix
            iteration_result['U'] = sherman_morisson.U

            # Sherman-Morisson b vector
            iteration_result['b'] = sherman_morisson.b

            # Sherman-Morisson u vector
            iteration_result['u'] = sherman_morisson.u

            # Sherman-Morisson v vector
            iteration_result['v'] = sherman_morisson.v

            # Sherman-Morisson x (solution) vector
            iteration_result['x'] = sherman_morisson.x

            iteration_results.append(iteration_result) 
        
        return iteration_results


    
    '''
        Returns statistics in the form of a Pandas dataframe.

        :param iteration_results: Iterations execution results.

        :returns: Pandas dataframe
    '''
    def print_statistics(self, iteration_results:[]):
        import pandas as pd

        iteration_stats = {
            'Μέγεθος προβλήματος (n)': [iteration.get('n') for iteration in iteration_results],
            'Μέθοδος οδήγησης': [iteration.get('pivoting') for iteration in iteration_results],
            'Χρόνος υπολογισμού (milliseconds)': [iteration.get('process_time') for iteration in iteration_results],
            'Δείκτης κατάστασης Α': [iteration.get('K_A') for iteration in iteration_results],
            'Νόρμα απείρου σφάλματος υπολογισμού': [iteration.get('e_c_inf_norm') for iteration in iteration_results],
            'Νόρμα απείρου σφάλματος διόρθωσης': [iteration.get('e_r_inf_norm') for iteration in iteration_results]
        }

        return pd.DataFrame(iteration_stats)



    '''
        Returns error statistics in the form of a Pandas dataframe.

        :param iteration_results: Iterations execution results.

        :returns: Pandas dataframe
    '''
    def print_error_statistics(self, iteration_results:[]):
        import pandas as pd

        error_stats = {
            'Μέγεθος προβλήματος (n)': [iteration.get('n') for iteration in iteration_results[0]],

            'Δείκτης κατάστασης Α': [iteration.get('K_A') for iteration in iteration_results[0]],
            
            'Νόρμα απείρου σφάλματος υπολογισμού (μερική οδήγηση)': [iteration.get('e_c_inf_norm') for iteration in iteration_results[0]],
            'Νόρμα απείρου σφάλματος υπολογισμού (ολική οδήγηση)': [iteration.get('e_c_inf_norm') for iteration in iteration_results[1]],
            'Διαφορά νόρμας απείρου σφάλματος υπολογισμού': np.array([iteration.get('e_c_inf_norm') for iteration in iteration_results[0]]) - np.array([iteration.get('e_c_inf_norm') for iteration in iteration_results[1]]),
            
            'Νόρμα απείρου σφάλματος διόρθωσης (μερική οδήγηση)': [iteration.get('e_r_inf_norm') for iteration in iteration_results[0]],
            'Νόρμα απείρου σφάλματος διόρθωσης (ολική οδήγηση)': [iteration.get('e_r_inf_norm') for iteration in iteration_results[1]],
            'Διαφορά νόρμας απείρου σφάλματος διόρθωσης': np.array([iteration.get('e_r_inf_norm') for iteration in iteration_results[0]]) - np.array([iteration.get('e_r_inf_norm') for iteration in iteration_results[1]])
        }

        return pd.DataFrame(error_stats)