#!/usr/bin/env python3
"""! @brief Python program for linear regression."""
##
# @file main.py
#
# @brief All homework 1 functions.
#
##
#
# @section Libraries/Modules
# - numpy extern library (https://numpy.org/)
# - matplotlib.pyplot extern library (https://matplotlib.org/stable/)
# - time standard library (https://docs.python.org/3/library/time.html)
#
# @section Auteur
# - PAULY Alexandre
##

# Imported library
import numpy as np
import matplotlib.pyplot as plt
import time

def read_data(file):
    """! Reading text file data function

    Function for reading and retrieving data from a file.

    @param file: File to read
    @type file: String (file path)

    @return: Coordinates tabs
    @rtype: Float table

    """

    # Loading data from a file
    donnees = np.loadtxt(file, delimiter=',')

    # Retrieving data
    x = donnees[:, 0]
    y = donnees[:, 1]

    return x, y

def design_matrix(x, n):
    """! Matrix design generation

    Function for generate the design matrix.

    @param x: Abscissa data
    @type x: Float tab

    @param n: Number of unknows
    @type n: Integer

    @return: Design matrix
    @rtype: Float Matrix

    """

    # Initialization of variables
    matrix = np.ones((len(x), 1))
    for i in range(1, n):
        matrix = np.column_stack([matrix, x ** i]) # Design matrix

    return matrix

def inverse_matrix(matrix):
    """! Matrix inversion

    Function for inverting a matrix by Gauss elimination method.

    @param matrix: Matrix
    @type matrix: Float matrix

    @return: Inverted matrix
    @rtype: Float matrix

    """

    # Initialization of variables
    n = len(matrix)                                        # Matrix length
    augmented_matrix = np.hstack([matrix, np.identity(n)]) # Augmented matrix
    
    # Gaussian elimination
    for i in range(n):
        pivot_row = i
        while augmented_matrix[pivot_row, i] == 0 and pivot_row < n:
            pivot_row += 1
        
        if pivot_row == n:
            raise ValueError("Matrix is singular and cannot be inverted.")
        
        # Swap rows
        augmented_matrix[[i, pivot_row]] = augmented_matrix[[pivot_row, i]]
        
        # Scale row to make pivot 1
        augmented_matrix[i] /= augmented_matrix[i, i]
        
        # Eliminate other rows
        for j in range(n):
            if j != i:
                augmented_matrix[j] -= augmented_matrix[j, i] * augmented_matrix[i]
    
    # Extract the inverse from the augmented matrix
    inverse = augmented_matrix[:, n:]
    
    return inverse

def calculate_error(y_pred, y):
    """! Error calculator

    Function used to calculate the error of a method.

    @param y_pred: Ordinate predicted data
    @type y_pred: Float tab

    @param y: Ordinate data
    @type y: Float tab

    @return: Error
    @rtype: Float

    """

    # Total error calculation
    error = np.sum((y_pred - y)**2)

    return error
    
def LSE_method(x, y, lambda_value, A):
    """! LSE method

    Function for regression using the LSE method.

    @param x: Abscissa data
    @type x: Float tab

    @param y: Ordinate data
    @type y: Float tab

    @param lambda_value: Lambda control parameters
    @type lambda_value: Integer

    @param A: Design matrix
    @type A: Float matrix

    @return: Ordinate predicted data
    @rtype: Float tab

    """

    # Initialising the time counter
    start = time.time() 

    # Gauss-Jordan elimination
    inverse_ATA_lambda = inverse_matrix(np.dot(np.transpose(A), A) + lambda_value * np.identity(n))

    # Calculation of regularization parameters
    theta = np.dot(np.dot(inverse_ATA_lambda, np.transpose(A)), y)

    # Prediction model
    y_pred = np.dot(A, theta)

    # Calculation of fitting line
    equation = " + ".join([f"{theta[i]:.10f} * x^{i}" for i in range(n)])

    # Calculation of total error
    error = calculate_error(y_pred, y)

    # Recording of treatment end time
    end = time.time()

    # Message displays
    print("\nLSE :")
    print(f"Fitting line : y_pred = {equation}")
    print(f"Total error : {error:.6f}")
    print(f"Execution time : {str(round(float(end-start),4))}")

    return y_pred

def newtons_method(x, y, A, ITER_MAX=1000, PRECISION=10**-5):
    """! Newton's method for regression

    Function for regression using Newton's method.

    @param x: Abscissa data
    @type x: Float tab

    @param y: Ordinate data
    @type y: Float tab

    @param lambda_value: Lambda control parameters
    @type lambda_value: Integer

    @param A: Design matrix
    @type A: Float matrix

    @param max_iterations: Maximum number of iterations
    @type max_iterations: Integer, optional

    @return: Ordinate predicted data
    @rtype: Float tab

    """

    # Initialization of variables
    start = time.time()          # Initialising the time counter
    theta = np.zeros(len(A[0]))  # Initialize parameters to zero
    i = 0                        # Loop index
    prev_theta = np.copy(theta)  # Store the previous theta for convergence check
    

    # While method's :
    # Advantages:
    #  - The process stops as soon as the criterion is reached.
    # Disadvantages:
    #  - If the stopping criterion is not chosen correctly, the process may become non-convergent.
    while i < ITER_MAX:
        gradient = 2 * np.dot(np.transpose(A), (np.dot(A, theta) - y))
        hessian = 2 * np.dot(np.transpose(A), A)
        theta -= np.linalg.solve(hessian, gradient)
        
        # Check for convergence
        if np.linalg.norm(theta - prev_theta) < PRECISION:
            break
        
        prev_theta = np.copy(theta)  # Update the previous theta
        i += 1

    # for method's :
    # Advantages:
    #  - Total control over the number of iterations.
    # Disadvantages:
    #  - Inefficient because the problem converges quickly, as there may be unnecessary iterations.
    # for _ in range(ITER_MAX):
    #     gradient = 2 * np.dot(np.transpose(A), (np.dot(A, theta) - y))
    #     hessian = 2 * np.dot(np.transpose(A), A)
    #     theta -= np.linalg.solve(hessian, gradient)

    # Prediction model
    y_pred = np.dot(A, theta)
    
    # Calculation of fitting line
    equation = " + ".join([f"{theta[i]:.10f} * x^{i}" for i in range(len(theta))])
    
    # Calculation of total error
    error = calculate_error(y_pred, y)

    # Recording of treatment end time
    end = time.time()
    
    # Message displays
    print("\nNewton's Method:")
    print(f"Fitting line : y_pred = {equation}")
    print(f"Total error : {error:.6f}")
    print(f"Execution time : {str(round(float(end-start),4))}")

    return y_pred

def plot(x, y, y_pred_lse, y_pred_newton):
    """! Plot display

    Function to display the plot of the two functions.

    @param x: Abscissa data
    @type x: Float tab

    @param y: Ordinate data
    @type y: Float tab

    @param y_pred_lse: Ordinate predicted data by LSE's method
    @type y_pred_lse: Float tab

    @param y_pred_newton: Ordinate predicted data by Newton's method
    @type y_pred_newton: Float tab

    """

    # Subplot initialization
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    # First subplot for LSE's method
    axs[0].scatter(x, y, label='Data')
    axs[0].plot(x, y_pred_lse, color='red', label='Regression Line')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].set_title('LSE\'s method')

    # Second subplot for Newton's method
    axs[1].scatter(x, y, label='Data')
    axs[1].plot(x, y_pred_newton, color='red', label='Regression Line')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].set_title('Newton\'s method')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Initialization of variables
    file_name = '../data/data.txt'                             # Data file name
    x, y = read_data(file_name)                                # Coordinate axes
    n = int(input('Enter number of unknowns : '))              # Number of unknows
    while n < 1:
        n = int(input('Enter positive number of unknowns : ')) # Number of unknows
    lambda_value = int(input('Enter value of lambda : '))      # Lambda control parameters
    A = design_matrix(x, n)                                    # Design matrix
    
    # LSE's method
    y_pred_lse = LSE_method(x, y, lambda_value, A)

    # Newton's method
    y_pred_newton = newtons_method(x, y, A)

    # Plot display
    plot(x, y, y_pred_lse, y_pred_newton)