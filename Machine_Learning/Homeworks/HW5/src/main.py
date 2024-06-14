#!/usr/bin/env python3
"""! @brief Python program for parameters generator."""
##
# @file main.py
#
# @brief All homework 5 functions.
#
##
#
# @section Libraries/Modules
# - libsvm extern library (https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
# - matplotlib.pyplot extern library (https://matplotlib.org/stable/)
# - numpy extern library (https://numpy.org/)
# - scipy extern library (https://scipy.org/)
#
# @section Auteur
# - PAULY Alexandre
##

# Imported library
from libsvm.svmutil import *
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize
import sys

def switch(exercise):
    """! Menu function

    Call function according to choice.

    @param exercice: Selected exercice
    @type exercice: Integer

    """

    if exercise == 1:
        # Initialization of hyperparameters
        beta = 5
        alpha = 1
        lengthscale = 1
        kernel_variance = 1
        predict_sample_size = 1000

        # Data initialization
        x, y = read_data("../data/input.data")

        # Gaussian process
        cov_matrix = covariance_matrix(x, beta, alpha, lengthscale, kernel_variance)                                                    # First step - Covariance matrix calculation
        mean , variance = gaussian_process_parameters(x, y, cov_matrix, beta, predict_sample_size, alpha, lengthscale, kernel_variance) # Second step - Parameters calculation
        plot_gaussian_process(x, y, mean, variance, predict_sample_size)                                                                # Third step - Plot results

        # Gaussian Process optimized by minimizing negative marginal log-likelihood
        opt = minimize(negative_marginal_log_likelihood, [alpha, lengthscale, kernel_variance], args = (x, y, beta))                                # Preprocessing - Calculation of optimized paameters
        alpha_opt, lengthscale_opt, kernel_variance_opt = opt.x
        cov_matrix = covariance_matrix(x, beta, alpha_opt, lengthscale_opt, kernel_variance_opt)                                                    # First step - Covariance matrix calculation with optimized parameters
        mean , variance = gaussian_process_parameters(x, y, cov_matrix, beta, predict_sample_size, alpha_opt, lengthscale_opt, kernel_variance_opt) # Second step - Parameters calculation with optimized hyperparameters
        plot_gaussian_process(x, y, mean, variance, predict_sample_size)                                                                            # Third step - Plot results
    elif exercise == 2: 
        # Initialization of variables
        X_train = read_csv("../data/X_train.csv", True)  # Training images
        Y_train = read_csv("../data/Y_train.csv", False) # Training labels
        X_test = read_csv("../data/X_test.csv", True)    # Test images
        Y_test = read_csv("../data/Y_test.csv", False)   # Test labels

        # Menu display
        print("\n1 - Linear, polynomial and RBF kernels")
        print("2 - C-SVC")
        print("3 - Linear kernel + RBD fernel together\n")

        method = int(input('Which method ? '))

        # Comparison between different kernel functions (linear, polynomial, and RBF kernels)
        if method == 1:
            print()
            for i in range(3):
                if i == 0:
                    print("Linear kernel :")
                elif i == 1:
                    print("Polynomial kernel :")
                elif i == 2:
                    print("RBF kernel :")
                model = svm_train(Y_train, X_train, f'-t {i} -q')
                result = svm_predict(Y_test, X_test, model)
        # C-SVC
        elif method == 2:
            original_stdout = sys.stdout

            # Open a file to store results
            with open('SVM.txt', 'w') as f:
                sys.stdout = f
                
                # For each kernel
                for i in range(4):
                    # Cross validation
                    optimal_option = grid_search(X_train, Y_train, i)
                    model = svm_train(Y_train, X_train, optimal_option)
                    result = svm_predict(Y_test, X_test, model)
                sys.stdout = original_stdout
        # Linear kernel + RBF kernel together
        elif method == 3:
            # Initialization of variables
            n_train = len(X_train) # Number of training images
            n_test = len(X_test)   # Number of test images

            # Kernel calculation of training and test images
            train_kernel = linear_kernel(X_train, X_train) + RBFkernel(X_train, X_train)
            test_kernel = linear_kernel(X_test, X_train) + RBFkernel(X_test, X_train)

            # Add index in front of kernel (Required for compatibility with the library)
            train_kernel = np.hstack((np.arange(1,n_train + 1).reshape(-1, 1), train_kernel))
            test_kernel = np.hstack((np.arange(1,n_test + 1).reshape(-1, 1), test_kernel))

            model = svm_train(Y_train, train_kernel, '-t 4 -q')
            result = svm_predict(Y_test, test_kernel, model)
        else:
            print("\nERROR : You have made no choice")

    else:
        print("\nERROR : You have made no choice")

def read_data(filepath):
    """! Reading text file data function

    Function for reading and retrieving data from a file.

    @param filepath: File to read
    @type filepath: String (file path)

    @return: abscissa and ordinate (X and y)
    @rtype: Array and Array

    """
        
    # Loading data from a file
    donnees = np.loadtxt(filepath, delimiter=' ')

    # Retrieving data
    x = donnees[:, 0]
    y = donnees[:, 1]

    return x, y

def read_csv(filepath, case):
    """! Reading csv file data function

    Function for reading and retrieving data from a file.

    @param filepath: File to read
    @type filepath: String (file path)

    @param case: if it's an image or a label file
    @type case: Boolean

    @return: Images or labels
    @rtype: 2D Array or Array

    """

    # Initialization of a list to store data
    if case: data = np.empty((0,784))
    else: data = []

    # Loading data from a file
    with open(filepath) as f:
        for line in f.readlines():
            if case:
                image = line.split(',')
                image = np.array(image).astype(float)
                data = np.vstack((data, image))
            else: 
                label = float(line)
                data.append(label)
    f.close()

    return data

def rational_quadratic_kernel(xi, xj, alpha, lengthscale, kernel_variance):
    """! Rational quadratic kernel calculation function

    Function to calculate the rational quadratic kernel.

    @param xi: i-th value of the dataset
    @type xi: Float

    @param xj: j-th value of the dataset
    @type xj: Float

    @param alpha: Alpha value
    @type alpha: Integer

    @param lengthscale: Lengthscale value
    @type lengthscale: Integer

    @param kernel_variance: Kernel variance
    @type kernel_variance: Integer

    @return: kernel value between i and j
    @rtype: Float

    """
        
    k = kernel_variance * (1 + ((xi-xj) ** 2) / (2 * alpha * lengthscale**2)) ** (-alpha)

    return k

def covariance_matrix(data, beta, alpha, lengthscale, kernel_variance):
    """! Covariance matrix creation function

    Function to calculate the covariance matrix of the sample data.

    @param data: Abscissa
    @type data: Float array

    @param beta: Beta value
    @type beta: Integer

    @param alpha: Alpha value
    @type alpha: Integer

    @param lengthscale: Lengthscale value
    @type lengthscale: Integer

    @param kernel_variance: Kernel variance
    @type kernel_variance: Integer

    @return: Covariance matrix
    @rtype: 2D Array

    """

    # Initialization of variable
    n = len(data)                 # Number of data
    cov_matrix = np.zeros((n, n)) # Covariance matrix

    # For each row
    for i in range(n):
        # For each column
        for j in range(n):
            cov_matrix[i][j] = rational_quadratic_kernel(data[i], data[j], alpha, lengthscale, kernel_variance)

            # On the diagonal, add 1/beta
            if i == j:
                cov_matrix[i][j] += 1 / beta

    return cov_matrix

def gaussian_process_parameters(data, target, cov_matrix, beta, predict_sample_size, alpha, lengthscale, kernel_variance):
    """! Parameters calculation function

    Function to calculate the parameters of the gaussian process method.

    @param data: Abscissa
    @type data: Float array

    @param target: Ordinate
    @type target: Float array

    @param cov_matrix: Covariance matrix
    @type cov_matrix: 2D Array

    @param beta: Beta value
    @type beta: Integer

    @param predict_sample_size: Number of point to predict
    @type predict_sample_size: Integer

    @param alpha: Alpha value
    @type alpha: Integer

    @param lengthscale: Lengthscale value
    @type lengthscale: Integer

    @param kernel_variance: Kernel variance
    @type kernel_variance: Integer

    @return: Mean and Variance
    @rtype: Array and Array

    """
        
    # Initialization of variable
    n = len(data)                                        # Number of data
    mean = np.zeros(predict_sample_size)                 # Mean
    variance = np.zeros(predict_sample_size)             # Variance
    x_sample = np.linspace(-60, 60, predict_sample_size) # Datapoints creation

    for sample in range(predict_sample_size):
        # Kernel initialization
        kernel = np.zeros((n, 1))
        for i in range(n):
            kernel[i][0] = rational_quadratic_kernel(data[i], x_sample[sample], alpha, lengthscale, kernel_variance)

        kernel_star = rational_quadratic_kernel(x_sample[sample], x_sample[sample], alpha, lengthscale, kernel_variance) + 1 / beta

        # Parameters calculation    
        mean[sample] = np.dot(np.dot(kernel.T, inv(cov_matrix)), target)
        variance[sample] = kernel_star - np.dot(np.dot(kernel.T, inv(cov_matrix)), kernel)

    return mean, variance

def negative_marginal_log_likelihood(theta, data, target, beta):
    """! Parameters calculation function

    Function to calculate the optimized parameters of the gaussian process method.

    @param data: Abscissa
    @type data: Float array

    @param target: Ordinate
    @type target: Float array

    @param cov_matrix: Covariance matrix
    @type cov_matrix: 2D Array

    @param beta: Beta value
    @type beta: Integer

    @param predict_sample_size: Number of point to predict
    @type predict_sample_size: Integer

    @param alpha: Alpha value
    @type alpha: Integer

    @param lengthscale: Lengthscale value
    @type lengthscale: Integer

    @param kernel_variance: Kernel variance
    @type kernel_variance: Integer

    @return: Mean and Variance
    @rtype: Array and Array

    """
    
    # Initialization of variables
    alpha = theta[0]           # Alpha
    lengthscale = theta[1]     # Lengthscale
    kernel_variance = theta[2] # Kernel variance
    n = len(data)              # Number of data

    # Negative marginal log likelihood
    cov_matrix_theta = covariance_matrix(data, beta, alpha, lengthscale, kernel_variance)
    likelihood = np.log(np.linalg.det(cov_matrix_theta)) / 2
    likelihood += np.dot(np.dot(target.T, inv(cov_matrix_theta)), target) / 2
    likelihood += n / 2 * np.log(2 * np.pi)

    return likelihood

def plot_gaussian_process(x, y, mean, variance, predict_sample_size):
    """! Plot display

    Function to display the plot of Gaussian process method.

    @param x: Abscissa data
    @type x: Float Array

    @param y: Ordinate data
    @type y: Float Array

    @param mean: Mean
    @type mean: Float Array

    @param variance: Variance
    @type variance: Float Array

    @param predict_sample_size: Number of point to predict
    @type predict_sample_size: Integer

    """

    # Line to represent mean of f in range [-60,60]
    x_sample = np.linspace(-60, 60, predict_sample_size)

    # Confidence interval of f
    interval = 1.96 * (variance ** 0.5) # 1.96 by reading law table of normal law

    # Plot display
    plt.scatter(x, y, color = 'k')
    plt.plot(x_sample, mean, color = 'b')
    plt.plot(x_sample, mean + interval, color = 'r')
    plt.plot(x_sample, mean - interval, color = 'r')
    plt.fill_between(x_sample, mean + interval, mean - interval, color = 'pink', alpha = 0.3)
    plt.show()

def best_option(data, target, option, optimal_option, optimal_accuracy):
    """! Best option

    Function to choose the best option.

    @param data: Data
    @type data: 2D Array

    @param target: Labels
    @type target: Array

    @param option: svm_train option
    @type option: String

    @param optimal_option: Optimal option
    @type optimal_option: String

    @param optimal_accuracy: Optimal accuracy
    @type optimal_accuracy: String

    """
        
    # SVM algo
    accuracy = svm_train(target, data, option)

    # Search for greater accuracy
    if accuracy > optimal_accuracy:
        return option, accuracy
    else:
        return optimal_option, optimal_accuracy

def grid_search(data, target, kernel):
    """! Best option

    Search function for finding parameters of best performing model.

    @param data: Data
    @type data: 2D Array

    @param target: Labels
    @type target: Array

    @param kernel: Kernel type (1, 2, 3 or 4)
    @type kernel: Integer

    @return: Optimal option
    @rtype: String

    """
        
    # Initialization of variables
    cost = [0.001, 0.01, 0.1, 1, 10, 100] # Costs value to test
    optimal_option = f'-s 0 -v 3 -q'      # Options 
    optimal_accuracy = 0                  # Optimal accuracy

    # If it's linear
    if kernel == 0:
        print('\nLinear kernel:')
        for c in cost:
            # Options initialization for svm_train
            option = f'-s 0 -t 0 -c {c} -q -v 3'
            print(option)

            # Cross validation
            optimal_option, optimal_accuracy = best_option(data, target, option, optimal_option, optimal_accuracy)
    # If it's polynomial
    elif kernel == 1:
        # Initialization of variables
        gamma = [0.0001, 1/784, 0.01, 0.1, 1, 10] # Gamma values to test
        coefficient = [-10, -5, 0, 5, 10]         # Coefficients in kernel
        degree = [1, 2, 3, 4]                     # Degree for polynomial kernel

        print('\nPolynomial kernel:')
        for c in cost:
            for d in degree:
                for g in gamma:
                    for r in coefficient:
                        # Options initialization for svm_train
                        option = f'-s 0 -t 1 -c {c} -d {d} -g {g} -r {r} -q -v 3'
                        print(option)

                        # Cross validation
                        optimal_option, optimal_accuracy = best_option(data, target, option, optimal_option, optimal_accuracy)
    # If it's RBF
    elif kernel == 2:
        # Initialization of variables
        gamma = [0.0001, 1/784, 0.01, 0.1, 1, 10] # Gamma values to test

        print('\nRBF kernel:')
        for c in cost:
            for g in gamma:
                # Options initialization for svm_train
                option = f'-s 0 -t 2 -c {c} -g {g} -q -v 3'
                print(option)

                # Cross validation
                optimal_option, optimal_accuracy = best_option(data, target, option, optimal_option, optimal_accuracy)
    # # If it's sigmoid
    elif kernel == 3:
        # Initialization of variables
        gamma = [0.0001, 1/784, 0.01, 0.1, 1, 10] # Gamma values to test
        coefficient = [-10, -5, 0, 5, 10]         # Coefficients in kernel

        print('\nSigmoid kernel:')
        for c in cost:
            for g in gamma:
                for r in coefficient:
                    # Options initialization for svm_train
                    option = f'-s 0 -t 3 -c {c} -g {g} -r {r} -q -v 3'
                    print(option)

                    # Cross validation
                    optimal_option, optimal_accuracy = best_option(data, target, option, optimal_option, optimal_accuracy)

    optimal_option = optimal_option[:-5]

    # Results display
    print(optimal_accuracy)
    print(optimal_option)

    return optimal_option

def linear_kernel(u, v):
    """! Linear_kernel

    Function to calculate the linear kernel.

    @param u: Matrix 1
    @type u: 2D Array

    @param v: Matrix 2
    @type v: 2D Array

    @return: linear_kernel(u,v)
    @rtype: Array and Array

    """
        
    return u @ v.T

def RBFkernel(u, v):
    """! RBF kernel

    Function to calculate the RBF kernel.

    @param u: Matrix 1
    @type u: 2D Array

    @param v: Matrix 2
    @type v: 2D Array

    @return: RBFkernel(u,v)
    @rtype: Array and Array

    """
    
    gamma = 1 / u.shape[1]
    dist = np.sum(u ** 2, axis=1).reshape(-1, 1) + np.sum(v ** 2, axis=1) - 2 * u @ v.T

    return np.exp(-gamma * dist)

if __name__ == "__main__":
    # Menu display
    print("\n################ MENU ################\n")
    print("1 - Gaussian process")
    print("2 - SVM on MNIST dataset\n")
    print("######################################\n")

    choice = int(input('Which exercice ? '))
    switch(choice)  