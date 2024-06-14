#!/usr/bin/env python3
"""! @brief Python program for parameters generator."""
##
# @file main.py
#
# @brief All homework 3 functions.
#
##
#
# @section Libraries/Modules
# - matplotlib.pyplot extern library (https://matplotlib.org/stable/)
# - numpy extern library (https://numpy.org/)
#
# @section Auteur
# - PAULY Alexandre
##

# Imported library
import matplotlib.pyplot as plt
import numpy as np


def switch(exercice):
    """! Menu function

    Call function according to choice.

    @param exercice: Selected exercice
    @type exercice: Integer

    """

    if exercice == 1:
        # Menu display
        print("\n1 - Univariate gaussian data generator")
        print("2 - Polynomial basis linear model data generator\n")

        question = int(input('Which question ? '))

        if question == 1:
            # Distribution parameters
            mean = float(input('\nWhich mean ? '))       # Mean
            variance = float(input('Which variance ? ')) # Variance
            while variance <= 0:
                variance = float(input('Which variance (must be positive) ? '))
            size = int(input('Which size ? '))         # Size
            while size <= 0:
                size = int(input('Which size (must be positive) ? '))

            # Message display
            print(f"\nData point source function: N({mean},{variance}) :")

            # Generation of gaussian data
            data = univariate_gaussian_data_generator(mean, variance, size)

            # Plot display
            plot(data, mean, variance, "Histogram of univariate gaussian data generator")
        elif question == 2:
            # Distribution parameters
            W = np.array([1, 2, 3, 4])                # Weights
            n = int(input('\nWhich basis number ? ')) # Basis number
            while n <= 0:
                n = int(input('Which basis number (must be positive) ? '))
            a = float(input('Which variance ? '))     # Variance
            while a <= 0:
                a = float(input('Which variance (must be positive) ? '))
            
            # Call function
            x, y = polynomial_basis_linear_model_data_generator(n, a, W)
            print(f"Data point : ({x}),({y})")
        else: 
            print("\nERROR : You have made no choice")
    elif exercice == 2:
        # Distribution parameters
        mean = float(input('\nWhich mean ? '))       # Mean
        variance = float(input('Which variance ? ')) # Variance
        while variance <= 0:
            variance = float(input('Which variance (must be positive) ? '))
        size = int(input('Which size ? '))         # Size
        while size <= 0:
            size = int(input('Which size (must be positive) ? '))

        # Univariate gaussian data generation to search estimator by sequential method
        data = univariate_gaussian_data_generator(mean, variance, size)

        # Message display
        print(f"\nData point source function: N({mean},{variance}) :")

        # Sequential method to search estimator
        estimate_mean, estimate_variance = sequential_estimator(data)

        # Plots display
        double_plot(data, mean, variance, estimate_mean, estimate_variance, "Histogram by sequential estimator")
    elif exercice == 3:
        # Distribution parameters
        b = int(input('\nWhich precision ? '))        # Precision
        while b <= 0:
            b = int(input('Which precision (must be positive) ? '))
        n = int(input('Which number of features ? ')) # Number of features
        while n <= 0:
            n = int(input('Which number of features (must be positive) ? '))
        a = int(input('Which variance ? '))           # Variance
        while a <= 0:
            a = int(input('Which variance (must be positive) ? '))
        if n == 3:
            W = np.array([1, 2, 3])                   # Weights
        else:
            W = np.array([1, 2, 3, 4])

        bayesian_linear_regression(b, n, a, W)
    else:
        print("\nERROR : You have made no choice")

def univariate_gaussian_data_generator(mean, variance, size):
    """! Univariante gaussian data generator

    Function to generate data by univariate gaussian method.

    @param mean: True mean
    @type mean: Float

    @param variance: True variance
    @type variance: Float

    @param size: Size
    @type size: Integer

    @return: Data calculted
    @rtype: Array

    """
        
    # Initialization of variables
    U = np.random.random(size) # Random values
    V = np.random.random(size) # Random values
    
    # Random variable calculation (Box-Muller formula)
    X = np.sqrt(-2 * np.log(U)) * np.cos(2 * np.pi * V)

    # Mean and variance adjustment
    data = mean + np.sqrt(variance) * X
    
    return data

def polynomial_basis_linear_model_data_generator(n, a, W):
    """! Polynomial basis linear model data generator

    Function to generate data by polynomial basis linear model method.

    @param n: Basis number
    @type n: Integer

    @param a: Variance
    @type a: Float

    @param W: Array
    @type W: Array

    @return: Abscissa of a point
    @rtype: Float

    @return: Ordinate of a point
    @rtype: Float

    """
        
    # Gaussian noise generation
    e = univariate_gaussian_data_generator(0, a, 1)

    # Abscissa with uniform distribution
    x = np.random.uniform(-1, 1)

    # Y calculation
    y = e
    for i in range(n):
        y = y + W[i] * (x ** i)

    return x, y

def sequential_estimator(data):
    """! Sequential estimator method

    Function to search estimators by sequential method.

    @param data: Data
    @type data: Array

    @return: Estimated mean
    @rtype: Float

    @return: Estimated variance
    @rtype: Float

    """
        
    # Initialization of variables
    n = 0           # Index
    mean = 0        # Mean
    mean_square = 0 # Squared mean
    variance = 0    # Variance

    # For each data point
    for x in data:
        n += 1                               # Index update
        delta = x - mean                     # Distance between x and the mean
        mean += delta / n                    # Mean update
        delta2 = x - mean                    # Distance between x and the new mean
        mean_square += delta * delta2        # Squared mean update
        # If n > 1 to avoid division by 0
        if n > 1:
            variance = mean_square / (n - 1) # Variance update

        # Messages displays
        print(f"Add data point: {x}")
        print(f"Mean = {mean}")
        if n > 1:
            print(f"Variance = {variance}")
        else:
            print("Variance = 0.0")

    return mean, variance

def update_posterior(x, y, posterior_mean, posterior_cov, a, n):
    """! Posterior update

    Function to update the posterior.

    @param x: Abscissa
    @type x: Array

    @param y: Ordinate
    @type y: Array

    @param posterior_mean: Posterior mean
    @type posterior_mean: Float

    @param posterior_cov: Posterior variance
    @type posterior_cov: Float

    @param a: Variance
    @type a: Float

    @param n: Number of features
    @type n: Integer

    @return: Posterior mean
    @rtype: Float

    @return: Posterior variance
    @rtype: Float

    """

    # Design matrix
    phi_x = np.array([x**i for i in range(n)])

    # Calculation to simplify the formula
    cov_inv = np.linalg.inv(posterior_cov)

    # Posterior calculation
    posterior_cov = np.linalg.inv(cov_inv + np.outer(phi_x, phi_x)/a)
    posterior_mean = np.dot(posterior_cov, np.dot(cov_inv, posterior_mean) + y*phi_x/a)

    return posterior_mean, posterior_cov

def bayesian_linear_regression(b, n, a, W):
    # Initializations of variables
    max_points = 500                   # Maximum data points
    x_observed = []                    # Abscissa observed
    y_observed = []                    # Ordinate observed
    posterior_mean = np.zeros(n)       # Mean of the posterior
    posterior_cov = np.identity(n) / b # Cov of the posterior

    # For each points
    for i in range(max_points):
        # Data coordinates initialization
        x, y = polynomial_basis_linear_model_data_generator(n, a, W)
        x_observed.append(x)
        y_observed.append(y)

        # Posterior update
        posterior_mean, posterior_cov = update_posterior(x, y, posterior_mean, posterior_cov, a, n)

        # Prediction calculation
        phi_x = np.array([x**i for i in range(len(posterior_mean))])
        predictive_mean = np.dot(posterior_mean, phi_x)
        predictive_var = a + np.dot(np.dot(phi_x, posterior_cov), phi_x.T)

        # Print results
        print("===========================================\n")
        print(f"At time that have seen {i} data points:\n")
        # print(f"Add data point ({x:.5f}, {y:.5f}):\n")
        print(f"Posterior mean: ")
        for mean in posterior_mean:
            print("    ",mean)
        print("\nPosterior covariance:")
        for row in posterior_cov:
            print("    ", end="")
            for cov in row:
                if cov < 0 :
                    print(f"{cov:.5f}", end=", ")
                else:
                    print(f" {cov:.5f}", end=", ")
            print()
        print(f"\nPredictive distribution ~ N({predictive_mean:.5f}, {predictive_var:.5f})\n")

    # Plot display
    bayesian_plot(x_observed, y_observed, posterior_mean, posterior_cov, a)

def plot(data, mean, variance, message):
    """! Plot display

    Function to display the plot of the two functions.

    @param data: Data
    @type data: Array

    @param mean: Real mean
    @type mean: Float

    @param variance: Real variance
    @type variance: Float

    @param message: Title of the plot
    @type message: String

    """
        
    # Histogram
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g')

    # Plot the probability density function of the Gaussian distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, len(data))
    p = (1 / (np.sqrt(2 * np.pi * variance))) * np.exp(-0.5 * ((x - mean) / np.sqrt(variance))**2)
    
    # Plot display
    plt.plot(x, p, 'k', linewidth=2)
    plt.title(message)
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.show()

def double_plot(data, mean, variance, estimate_mean, estimate_variance, message):
    """! Double plot display

    Function to display a plot with two subplot. One with the true parameters and one with estimated parameters.

    @param data: Data
    @type data: Array

    @param mean: Real mean
    @type mean: Float

    @param variance: Real variance
    @type variance: Float

    @param estimate_mean: Estimated mean
    @type estimate_mean: Float

    @param estimate_variance: Estimated variance
    @type estimate_variance: Float

    @param message: Title of the estimated plot
    @type message: String

    """
        
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot and histogram displays
    for ax, est_mean, est_variance, est_label in zip(axs, [mean, estimate_mean], [variance, estimate_variance], ['Actual', 'Estimated']):
        ax.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Data Histogram')
        
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, len(data))
        p = (1 / (np.sqrt(2 * np.pi * est_variance))) * np.exp(-0.5 * ((x - est_mean) / np.sqrt(est_variance))**2)
        ax.plot(x, p, 'k', linewidth=2, label='Gaussian')

        ax.set_title(f'{est_label} Mean = {est_mean:.2f} and Variance = {est_variance:.2f}')
        ax.set_xlabel('Values')
        ax.set_ylabel('Density')
        ax.legend()
    plt.suptitle(message)
    plt.show()

def bayesian_plot(x_observed, y_observed, posterior_mean, posterior_cov, a):
    """! Plot display

    Function to display the plot of the bayesian linear regression.

    @param x: Abscissa observed
    @type x: Array

    @param y: Ordinate observed
    @type y: Array

    @param posterior_mean: Posterior mean
    @type posterior_mean: Float

    @param posterior_cov: Posterior variance
    @type posterior_cov: Float

    @param a: Variance
    @type a: Float

    """
        
    # Plots display
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x_range = np.linspace(-2, 2, 1000)
    predictive_means = []
    predictive_upper = []
    predictive_lower = []
    for x_val in x_range:
        # Prediction calculation
        phi_x = np.array([x_val**i for i in range(len(posterior_mean))])
        predictive_mean = np.dot(posterior_mean, phi_x)
        predictive_var = a + np.dot(np.dot(phi_x, posterior_cov), phi_x.T)
        
        predictive_means.append(predictive_mean)
        predictive_upper.append(predictive_mean + np.sqrt(predictive_var))
        predictive_lower.append(predictive_mean - np.sqrt(predictive_var))

    # Ground Truth
    axes[0, 0].plot(x_range, predictive_means, color='k', label='Mean')
    axes[0, 0].plot(x_range, predictive_upper, color='r', linestyle='--', label='Mean + Variance')
    axes[0, 0].plot(x_range, predictive_lower, color='r', linestyle='--', label='Mean - Variance')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].legend()

    # Predict Result
    axes[0, 1].scatter(x_observed, y_observed, color='b', label='Data Points')
    axes[0, 1].plot(x_range, predictive_means, color='k', label='Mean')
    axes[0, 1].plot(x_range, predictive_upper, color='r', linestyle='--', label='Mean + Variance')
    axes[0, 1].plot(x_range, predictive_lower, color='r', linestyle='--', label='Mean - Variance')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_title('Predict result')
    axes[0, 1].legend()

    # After 10 data points
    axes[1, 0].scatter(x_observed[:10], y_observed[:10], color='b', label='Data Points')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_title('After 10 incomes')
    axes[1, 0].plot(x_range, predictive_means, color='k', label='Mean')
    axes[1, 0].plot(x_range, predictive_upper, color='r', linestyle='--', label='Mean + Variance')
    axes[1, 0].plot(x_range, predictive_lower, color='r', linestyle='--', label='Mean - Variance')
    axes[1, 0].legend()

    # After 50 data points
    axes[1, 1].scatter(x_observed[:50], y_observed[:50], color='b', label='Data Points')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_title('After 50 incomes')
    axes[1, 1].plot(x_range, predictive_means, color='k', label='Mean')
    axes[1, 1].plot(x_range, predictive_upper, color='r', linestyle='--', label='Mean + Variance')
    axes[1, 1].plot(x_range, predictive_lower, color='r', linestyle='--', label='Mean - Variance')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Menu display
    print("\n################ MENU ################\n")
    print("1 - Random Data Generator")
    print("2 - Sequential Estimator")
    print("3 - Baysian Linear regression\n")
    print("######################################\n")

    choice = int(input('Which exercice ? '))
    switch(choice)    