#!/usr/bin/env python3
"""! @brief Python program for beta approximation."""
##
# @file main.py
#
# @brief All homework 2 functions.
#
##
#
# @section Libraries/Modules
# - gzip standard library (https://docs.python.org/3/library/gzip.html)
# - IPython.display extern library (https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html)
# - matplotlib.pyplot extern library (https://matplotlib.org/stable/)
# - numpy extern library (https://numpy.org/)
# - PIL extern library (https://he-arc.github.io/livre-python/pillow/index.html)
# - scipy.stats extern library (https://docs.scipy.org/doc/scipy/reference/stats.html)
#
# @section Auteur
# - PAULY Alexandre
##

# Imported library
from IPython.display import display
import gzip
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import binom

def switch(choice):
    """! Menu function

    Call function according to choice.

    @param choice: Choice
    @type choice: Integer

    """

    if choice == 1:
        print()
        naive_bayes_classifier()
    elif choice == 2:
        # Initialization of variables
        file_path = "../data/testfile.txt"                 # Data file
        a = int(input('\nEnter a parameter value : a = ')) # a parameter value
        b = int(input('Enter b parameter value : b = '))   # b parameter value
        print()

        online_learning(file_path, a, b)
    elif choice == 3:
        file_path = '../doc/beta_binomial_conjugation_proof.png'
        image = Image.open(file_path)
        image.show()
        display(image)
    elif choice == 4:
        print("\nOpen : beta_binomial_conjugation_proof.pdf in doc directory")
    else:
        print("\nERROR : You have made no choice")

def read_idx(file):
    """! Reading text file data function

    Function for reading and retrieving data from a file.

    @param file: File to read
    @type file: String (file path)

    @return: Data array
    @rtype: Array

    """

    # Loading data from a file
    with gzip.open(file, 'rb') as f:
        # Initialization of variables
        magic_number = int.from_bytes(f.read(4), 'big') # Magic number (to differentiate data)
        items = int.from_bytes(f.read(4), 'big')        # Number of items

        # Reading data as a function of the magic number
        if magic_number == 2051:
            # Initialization of variables
            rows = int.from_bytes(f.read(4), 'big')                                   # Number of rows
            cols = int.from_bytes(f.read(4), 'big')                                   # Number of cols
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(items, rows, cols) # Data to be processed (images)
        elif magic_number == 2049:
            # Initialization of variables
            data = np.frombuffer(f.read(), dtype=np.uint8)                            # Data to be processed (labels)
        else:
            raise ValueError(f"Unknown magic number: {magic_number}")

    return data

def split_pixels(image):
    """! Pixel spliting

    Function for split pixel in 32 bins for an image.

    @param image: Image to split in bins/cells
    @type image: Array(28,28)

    @return: Size of each cell
    @rtype: Array(32)

    """
        
    # Calculate the number of each pixel in 32 cells
    bin_counts, _ = np.histogram(image, bins=range(0, 257, 8))
    
    return bin_counts

def calculate_frequencies(bin_counts):
    """! Frequencies

    Function for calculate frequencies of each bins for each classes.

    @param bin_counts: Splited pixels i 32 bins for each classes
    @type bin_counts: Array(10, 32)

    @return: Frenquencies of each bisn for each classes
    @rtype: Array(10, 32)

    """
        
    # Initialization of variables
    frequencies = np.zeros_like(bin_counts) # Frequencies of each pixels of each classes
    
    # For each classes
    for i in range(10):
        # Initialization of variables
        class_counts = bin_counts[i]            # Frequency of this classe
        total_count = int(np.sum(class_counts)) # Total frequency of this classe

        # For each bins
        for j in range(32):
            # If empty bin
            if class_counts[j] == 0:
                min_count = np.min(class_counts[np.nonzero(class_counts)])
                frequencies[i][j] = min_count / total_count
            else:
                frequencies[i][j] = class_counts[j] / total_count

    return frequencies

def estimate_parameters(train_images, train_labels):
    """! Estimation parameters function

    Function for estimate parameters for each pixels of each labels

    @param train_images: Training images
    @type train_images: 3D Array(10, 28, 28)

    @param train_labels: Training labels
    @type train_labels: Array(10)

    @return: Array for the mean of each pixels of each classes
    @rtype: 3D Array(10, 28, 28)

    @return: Array for the standard error of each pixels of each classes
    @rtype: 3D Array(10, 28, 28)

    """
        
    # Initialization of variables
    means = np.zeros((10, 28, 28))           # Mean of each pixels of each classes
    standard_errors = np.zeros((10, 28, 28)) # Standard error of each pixels of each classes

    # For each classes
    for i in range(10):  # Pour chaque classe
        class_images = train_images[train_labels == i]    # Select images from the current class
        means[i] = np.mean(class_images, axis=0)          # Mean calculation for each pixel
        standard_errors[i] = np.std(class_images, axis=0) # Standard error calculation for each pixel

    return means, standard_errors

# def prior_proba(train_labels):
#     """! Prior calculation

#     Function for estimate prior probability for each classes

#     @param train_labels: Training labels
#     @type train_labels: Array

#     @return: Array for the prior
#     @rtype: Array(10)

#     """

#     # Initialization of variables
#     n = len(train_labels)                    # Number of images
#     prior_probabilities = np.zeros(10)       # Empty table for prior probability
#     class_counts = np.bincount(train_labels) # Calcul des occurrences de chaque classe dans les étiquettes d'entraînement
    
#     # Prior probablity calculation for each classes
#     for label in range(10):
#         prior_probabilities[label] = class_counts[label] / n
    
#     return prior_probabilities

def naive_bayes_discrete(test_images, test_labels, prior_probabilities, frequencies):
    """! Naives bayes in discrete mode

    Function for naives bayes classifer in discrete mode.

    @param test_images: Test images
    @type test_images: 3D Array(10, 28, 28)

    @param test_labels: Test labels
    @type test_labels: Array(10)

    @param prior_probabilities: Prior probabilities
    @type prior_probabilities: Array(10)

    @param frequencies: Frequencies of each casse for each bins
    @type frequencies: Array(10, 32)

    @return: Prediction
    @rtype: Integer array

    """

    # Initialization of variable
    predictions = np.zeros(len(test_images)) # Predictions list
    cpt = 0                                  # Loop index 

    # For each images
    for image in test_images:
        # Initialization of variable
        bin_counts = split_pixels(image)   # Calculate the number of each pixel in 32 cells for an image
        posterior_log_proba = np.zeros(10) # Array to store log posterior probabilities

        # For each classes
        for i in range(10):            
            # Posterior probability calculating
            posterior_log_proba[i] = np.sum(np.log(frequencies[i]) * bin_counts) + np.log(prior_probabilities[i])
        
        # Use of normalisation to avoid values that are too small (nan or 0)
        posterior_proba = np.exp(posterior_log_proba - np.max(posterior_log_proba))
        
        # Prediction of the higher probability
        predicted_class = np.argmax(posterior_proba)
        predictions[cpt] = predicted_class

        if cpt == 0 or cpt == 1:
            # Messages display
            print("\nPosterior (in log scale):")
            for l in range(10):
                print(f"{l}: ", posterior_proba[l] / sum(posterior_proba))
            print(f"Prediction: {int(predicted_class)}, Ans: {test_labels[cpt]}")

        cpt += 1

    return predictions

def naive_bayes_continuous(test_images, test_labels, prior_probabilities, means, variances):
    """! Naives bayes in continuous mode

    Function for naives bayes classifer in continuous mode.

    @param test_images: Test images
    @type test_images: 3D Array(10, 28, 28)

    @param test_labels: Test labels
    @type test_labels: Array(10)

    @param prior_probabilities: Prior probabilities
    @type prior_probabilities: Array(10)

    @param means: Array for the mean of each pixels of each classes
    @type means: 3D Array(10, 28, 28)

    @param std_error: Array for the standard error of each pixels of each classes
    @type std_error: 3D Array(10, 28, 28)

    @return: Prediction
    @rtype: Integer array

    """
        
    # Initialization of variable
    predictions = np.zeros(len(test_images)) # Predictions list
    
    # For each test images
    for k, image in enumerate(test_images):
        # Initialization of variable
        posterior_proba = np.zeros(10) # Posterior probability
        
        # Parcourir chaque classe
        for classe in range(10):
            # Initialization of variable
            posterior_log_proba = np.log(prior_probabilities[classe]) # Posterior probability
            
            # For each pixel
            for i in range(len(image)):
                for j in range(len(image[0])):
                    # Initialization of variables
                    pixel_value = image[i][j]          # Pixel value
                    mean = means[classe][i][j]         # Mean of this pixel
                    variance = variances[classe][i][j] # Variance of this pixel
                    
                    # If variance is very close than 0, we add a little smoothing to avoid division by 0
                    if variance == 0:
                        proba = -0.5 * ((pixel_value - mean) ** 2) / (variance + 1e-6)
                    else:
                        proba = -0.5 * ((pixel_value - mean) ** 2) / variance - 0.5 * np.log(2 * np.pi * variance)
                    
                    # Add of this proba to an array 
                    posterior_log_proba += proba

            # Add of this proba to an array for this classe
            posterior_proba[classe] = posterior_log_proba
        
        # Prediction of the higher probability
        prediction = np.argmax(posterior_proba)
        predictions[k] = prediction
        
        if k == 0 or k == 1:
            # Messages display
            print("\nPosterior (in log scale):")
            for l in range(10):
                print(f"{l}: ", posterior_proba[l]/sum(posterior_proba))
            print(f"Prediction: {int(predictions[k])}, Ans: {test_labels[k]}")
    
    return predictions

def naive_bayes_continuous2(test_images, test_labels, prior_probabilities, smoothing, means, std_error):
    """! Naives bayes in continuous mode

    Function for naives bayes classifer in continuous mode.

    @param test_images: Test images
    @type test_images: 3D Array(10, 28, 28)

    @param test_labels: Test labels
    @type test_labels: Array(10)

    @param prior_probabilities: Prior probabilities
    @type prior_probabilities: Array(10)

    @param smoothing: Smoothing value
    @type smoothing: Integer

    @param means: Array for the mean of each pixels of each classes
    @type means: 3D Array(10, 28, 28)

    @param std_error: Array for the standard error of each pixels of each classes
    @type std_error: 3D Array(10, 28, 28)

    @return: Prediction
    @rtype: Integer array

    """
    
    # Initialization of variables
    predictions = np.zeros(len(test_images)) # Predictions list
    
    # For each test images
    for n in range(len(test_images)):
        # Initialization of variables
        posterior_proba = [] # Posterior probabiliy list
        sample = test_images[n]      # Image
        
        # For each classe
        for i in range(10):
            # Initialization of variabless
            mean = means[i]                           # Mean
            var = np.square(std_error[i]) + smoothing # Variance with smoothing
            
            # Probability calculating
            proba = 1 / np.sqrt(2 * np.pi * var) * np.exp(-np.square(sample - mean)/(2 * var))
            
            # Classifier calculating
            result = np.sum(np.log(proba)) + np.log(prior_probabilities[i])
            posterior_proba.append(result)

        # Prediction of the higher probability
        predictions[n] = np.argmax(posterior_proba)

        if n == 0 or n == 1:
            # Messages display
            print("\nPosterior (in log scale):")
            for k in range(10):
                print(f"{k}: ", posterior_proba[k]/sum(posterior_proba))
            print(f"Prediction: {int(predictions[n])}, Ans: {test_labels[n]}")
    
    return predictions

def calculate_error_rate(predictions, test_labels):
    """! Calculation error rate function

    Function for calculing error rate from classification.

    @param predictions: Predicted value
    @type predictions: Integer array

    @param test_labels: True value
    @type test_labels: Integer array

    @return: Error rate
    @rtype: Float

    """
    
    # Error rate
    error_rate = np.sum(predictions != test_labels) / len(test_labels)
    
    return error_rate

def plot(means, display = True):
    """! Plots display

    Function for displaying the plot of each class for the number imagination.

    @param means: Means value of each pixels
    @type means: 2D array 

    @param display: True to display imagined numbers one by one and False to display all imagined numbers on the same plot
    @type display: Boolean

    """
        
    # Initialization of variables
    predicted_numbers = [] # List of number imagined
    
    # For each classes
    for classe in range(10):
        # Initialization of variables
        prediction_img = np.zeros((28, 28)) # New empty image for the number imagination
        
        # For each pixel of the image
        for i in range(28):
            for j in range(28):
                # If the mean value of the pixel is greater than or less than 128, it's black or white (0 or 1)
                if means[classe][i][j] < 128:
                    prediction_img[i][j] = 1
                else:
                    prediction_img[i][j] = 0
        
        # Add the imagined image to the list
        predicted_numbers.append(prediction_img)

    # Message display
    print("\nImagination of numbers in Bayesian classifier : \n")

    # Different way to display (a plot for each classes or a plot for all classes)
    if display:    
        # Display of imaginary numbers
        for label, number in enumerate(predicted_numbers):
            print(f"Imagined image for label {label}:")
            print(number)
            print()
            # plt.imshow(number, interpolation='nearest')
            # plt.show()
    else:
        fig, axes = plt.subplots(2, 5, figsize=(15, 7))
        
        # Display of imaginary numbers
        for label, number in enumerate(predicted_numbers):
            row = label // 5
            col = label % 5
            axes[row, col].imshow(number, interpolation='nearest')
            axes[row, col].set_title(f'Class {label}')
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

def naive_bayes_classifier():
    # Initialization of variables
    train_images = read_idx('../data/train-images-idx3-ubyte.gz')[:60000] # Training images
    train_labels = read_idx('../data/train-labels-idx1-ubyte.gz')[:60000] # Training labels
    test_images = read_idx('../data/t10k-images-idx3-ubyte.gz')[:10000]   # Test images
    test_labels = read_idx('../data/t10k-labels-idx1-ubyte.gz')[:10000]   # Test labels

    # Inverting pixel values
    train_images = 255 - train_images
    test_images = 255 - test_images

    # Asks the user to choose their method
    print("1 - Discrete mode")
    print("2 - Continuous mode (method 1)")
    print("3 - Continuous mode (method 2)")
    method = int(input('\nWhich mode ? '))

    # Switch to choose the method
    if method == 1 :
        # Initialization of variables
        bin_counts = np.zeros((10, 32)) # Empty array for tally the count of each pixel into 32 bins for each classes

        # Calculate the number of each pixel in 32 cells
        for i in range(len(train_images)):
            class_label = train_labels[i]
            bin_counts[class_label] += split_pixels(train_images[i])

        # Calculate the frequencie of each pixel in 32 cells
        frequencies = calculate_frequencies(bin_counts)
        
        # Prior calculation
        # prior_proba = prior_proba(train_labels)
        n = len(train_labels)                    # Number of images
        prior_probabilities = np.zeros(10)       # Empty table for prior probability
        class_counts = np.bincount(train_labels) # Calcul des occurrences de chaque classe dans les étiquettes d'entraînement
        
        # Prior probablity calculation for each classes
        for label in range(10):
            prior_probabilities[label] = class_counts[label] / n

        prior_proba = prior_probabilities

        # Classification
        predictions = naive_bayes_discrete(test_images, test_labels, prior_proba, frequencies)

        # Means and standards calculation
        means, variances = estimate_parameters(train_images, train_labels)

        # Plot of imagined numbers
        plot(means)

        # Error rate calculation
        error = calculate_error_rate(predictions, test_labels)
        print("Error rate : ", error)  
    elif method == 2:
        # Means and standards calculation
        means, variances = estimate_parameters(train_images, train_labels)

        # Prior calculation
        # prior_proba = prior_proba(train_labels)
        n = len(train_labels)                    # Number of images
        prior_probabilities = np.zeros(10)       # Empty table for prior probability
        class_counts = np.bincount(train_labels) # Calcul des occurrences de chaque classe dans les étiquettes d'entraînement
        
        # Prior probablity calculation for each classes
        for label in range(10):
            prior_probabilities[label] = class_counts[label] / n

        prior_proba = prior_probabilities

        # Classification
        predictions = naive_bayes_continuous(test_images, test_labels, prior_proba, means, variances)

        # Plot of imagined numbers
        plot(means)

        # Error rate calculation
        error = calculate_error_rate(predictions, test_labels)
        print("Error rate : ", error)  
    elif method == 3:
        # Means and standards calculation
        means, variances = estimate_parameters(train_images, train_labels)

        # Prior calculation
        # prior_proba = prior_proba(train_labels)
        n = len(train_labels)                    # Number of images
        prior_probabilities = np.zeros(10)       # Empty table for prior probability
        class_counts = np.bincount(train_labels) # Calcul des occurrences de chaque classe dans les étiquettes d'entraînement
        
        # Prior probablity calculation for each classes
        for label in range(10):
            prior_probabilities[label] = class_counts[label] / n

        prior_proba = prior_probabilities

        # Classification
        predictions = naive_bayes_continuous2(test_images, test_labels, prior_proba, 1000, means, variances)

        # Plot of imagined numbers
        plot(means)

        # Error rate calculation
        error = calculate_error_rate(predictions, test_labels)
        print("Error rate : ", error)
    else: 
        print("\nERROR : You have made no choice")

def online_learning(file_path, prior_a, prior_b):
    """! Online learning

    Function to make an online learning.

    @param file_path: File
    @type file_path: String

    @param prior_a: a parameter
    @type prior_a: Integer

    @param prior_b: b parameter
    @type prior_b: Integer

    """

    # For each line of the file
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, 1):
            # Initialization of variables
            line = line.strip()         # Line to be treated
            successes = line.count('1') # Number of 1
            failures = line.count('0')  # Number of 0
            
            # Binomial likelihood (MLE)
            likelihood = binom.pmf(successes, successes + failures, successes / (successes + failures))
            
            # Beta posterior parameters
            posterior_a = prior_a + successes
            posterior_b = prior_b + failures

            # Result displays
            print(f"case {line_number}: {line}")
            print(f"   Likelihood (Binomial MLE): {likelihood}")
            if prior_a < 10:
                print(f"   Beta Prior:     a = {prior_a},  b = {prior_b}")
            else: 
                print(f"   Beta Prior:     a = {prior_a}, b = {prior_b}")
            print(f"   Beta Posterior: a = {posterior_a}, b = {posterior_b}")
            print()

            # Update of values
            prior_a = posterior_a
            prior_b = posterior_b

if __name__ == "__main__":
    # Menu display
    print("\n################ MENU ################\n")
    print("1 - Naive Bayes classifier")
    print("2 - Online learning")
    print("3 - Prove Beta-Binomial conjugation (handwritten version)")
    print("4 - Prove Beta-Binomial conjugation (latex version : more details)\n")
    print("######################################\n")

    choice = int(input('Which exercice ? '))
    switch(choice)      