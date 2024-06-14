#!/usr/bin/env python3
"""! @brief Python program for parameters generator."""
##
# @file main.py
#
# @brief All homework 4 functions.
#
##
#
# @section Libraries/Modules
# - gzip standard library (https://docs.python.org/3/library/gzip.html)
# - matplotlib.pyplot extern library (https://matplotlib.org/stable/)
# - numpy extern library (https://numpy.org/)
#
# @section Auteur
# - PAULY Alexandre
##

# Imported library
import gzip
import matplotlib.pyplot as plt
import numpy as np

def switch(exercice):
    """! Menu function

    Call function according to choice.

    @param exercice: Selected exercice
    @type exercice: Integer

    """

    if exercice == 1:
        # Initialization of variables
        test = 2
        N = 50                       # Number of data for each group
        mx1 = my1 = 1                # Mean of the first group
        if test == 1: mx2 = my2 = 10 # Mean of the second group
        else: mx2 = my2 = 3
        vx1 = vy1 = 2                # Variance of the first group
        if test == 1: vx2 = vy2 = 2  # Variance of the second group
        else: vx2 = vy2 = 4

        # Initialization of data for the logistic regression
        X, y = init_logistic_reg(N, mx1, my1, mx2, my2, vx1, vy1, vx2, vy2)

        # Logistic Regression
        if test == 1: logistic_regression(X, y, 1e-4)
        else: logistic_regression(X, y, 1e-3)
    elif exercice == 2:
        # Initialization of variables
        train_images = read_idx('../data/train-images-idx3-ubyte.gz')[:1000] # Training images
        train_labels = read_idx('../data/train-labels-idx1-ubyte.gz')[:1000] # Training labels
        
        # Shape transformation
        train_images = train_images.reshape(train_images.shape[0], -1)

        em_algorithm(train_images, train_labels)
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
    
    # Random variable calculation
    X = np.sqrt(-2 * np.log(U)) * np.cos(2 * np.pi * V)

    # Mean and variance adjustment
    data = mean + np.sqrt(variance) * X
    
    return data

def init_logistic_reg(N, mx1, my1, mx2, my2, vx1, vy1, vx2, vy2):
    """! Logistic regression data initialization

    Function to initialize data by univariate gaussian method for the logistic regression.

    @param N: Number of data points
    @type N: Integer

    @param mx1: Mean for the first abscissa vector
    @type mx1: Float

    @param my1: Mean for the first ordinate vector
    @type my1: Float

    @param mx2: Mean for the second abscissa vector
    @type mx2: Float

    @param my2: Mean for the second ordinate vector
    @type my2: Float

    @param vx1: Variance for the first abscissa vector
    @type vx1: Float

    @param vy1: Variance for the first ordinate vector
    @type vy1: Float

    @param vx2: Variance for the second abscissa vector
    @type vx2: Float

    @param vy2: Variance for the second ordinate vector
    @type vy2: Float

    @return: Data generated
    @rtype: 2D Array

    @return: Labels generated
    @rtype: Array

    """
        
    # Initialization of variables
    vect1_x = univariate_gaussian_data_generator(mx1, vx1, N) # x1, ..., xn abscissa with N(mx1,vx1)
    vect1_y = univariate_gaussian_data_generator(my1, vy1, N) # y1, ..., yn ordinate with N(my1,vy1)
    vect2_x = univariate_gaussian_data_generator(mx2, vx2, N) # x1, ..., xn abscissa with N(mx2,vx2)
    vect2_y = univariate_gaussian_data_generator(my2, vy2, N) # y1, ..., yn ordinate with N(my2,vy2)
    D1_labels = np.zeros(N)                                   # Labels for D1
    D2_labels = np.ones(N)                                    # Labels for D2

    # Vectors initialization
    D1 = np.hstack((vect1_x, vect2_x)) # Data points with (x1,y1), ..., (xn,yn) with N(mx1,vx1) and N(my1,vy1)
    D2 = np.hstack((vect1_y, vect2_y)) # Data points with (x1,y1), ..., (xn,yn) with N(mx2,vx2) and N(my2,vy2)

    # Dataset
    data = np.vstack((D1, D2)).T
    
    # Classes initialization
    labels = np.concatenate((D1_labels, D2_labels))

    return data, labels

def sigmoid(z):
    """! Sigmoid function for logistic regression

    Function to Calculate the sigmoid value in z = np.dot(X,theta).

    @param X: Design matrix
    @type X: Array

    @param theta: Features
    @type theta: Array

    @return: Predictions
    @rtype: Array
    print(train_images.shape)
    """

    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    """! Cost function for logistic regression

    Function to calculate the loss.

    @param X: Design matrix
    @type X: Array

    @param y: Labels
    @type y: Array

    @param theta: Features
    @type theta: Array

    @return: Cost
    @rtype: Float

    """

    # Initialization of variables
    n = len(y)                       # Number of data
    h = sigmoid(np.dot(X, theta))    # Sigmoid function for the logistic regression
    h = np.clip(h, 1e-15, 1 - 1e-15) # To not divided by 0

    # Loss calcul
    loss = (-1 / n) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

    return loss

def gradient_descent(X, y, learning_rate, max_iter, epsilon):
    """! Gradient descent

    Function to calculate weights with the gradient's method.

    @param X: Design matrix
    @type X: Array

    @param y: Labels
    @type y: Array

    @param weights: Features
    @type weights: Array

    @param learning_rate: Learning rate
    @type learning_rate: Float

    @param max_iter: Number of iterations
    @type max_iter: Integer

    @param epsilon: Stop criteria
    @type epsilon: Float

    @return: Optimized weights
    @rtype: Arrayprint(train_images.shape)

    """

    # Initialization of variables
    design_matrix = np.hstack((np.ones((X.shape[0], 1)), X)) # Design matrix
    weights = np.zeros(design_matrix.shape[1])               # Weights initialization
    n = len(y)                                               # Number of data
    cost = cost_function(design_matrix, y, weights)          # Cost 

    # For each iterations
    for _ in range(max_iter):
        # Sigmoid function for the logistic regression
        h = sigmoid(np.dot(design_matrix, weights))

        # Error calculation
        error = h - y

        # Gradient
        gradient = np.dot(design_matrix.T, error)

        # Weights update
        weights -= learning_rate * gradient

        # New cost
        new_cost = cost_function(design_matrix, y, weights)

        # Stop criteria
        if abs(cost - new_cost) < epsilon:
            break

        # Cost update
        cost = new_cost

    return weights

def newtons_method(X, y, max_iter, epsilon=1e-5):
    """! Newton's method

    Function to calculate weights with the Newton's method.

    @param X: Design matrix
    @type X: Array

    @param y: Labels
    @type y: Array

    @param weights: Features
    @type weights: Array

    @param max_iter: Number of iterations
    @type max_iter: Integer

    @param epsilon: Stop criteria
    @type epsilon: Float

    @return: Optimized weights
    @rtype: Array

    """

    # Initialization of variables
    design_matrix = np.hstack((np.ones((X.shape[0], 1)), X)) # Design matrix
    weights = np.zeros(design_matrix.shape[1])               # Weights initialization
    n = len(y)                                               # Number of data

    # For each iterations
    for _ in range(max_iter):
        # Sigmoid function for the logistic regression
        h = sigmoid(np.dot(design_matrix, weights))  
        
        # Error calculation
        error = h - y      

        # Gradient and hessian                                          
        gradient = np.dot(design_matrix.T, error)
        hessian = np.dot(design_matrix.T, np.dot(np.diag(h), np.dot(np.diag(1 - h), design_matrix)))

        # If hessian isn't singular
        try:
            delta_weights = np.linalg.inv(hessian).dot(gradient)
        # Else if hessian is singular, resort to gradient descent
        except np.linalg.LinAlgError:
            delta_weights = gradient_descent(X, y, learning_rate=0.01, max_iter=max_iter, epsilon=epsilon) # Weights with gradient's method
            break
        
        # Weights update
        weights -= delta_weights

        # Stop criteria
        if np.linalg.norm(delta_weights) < epsilon:
            break
        
    return weights

def confusion_matrix_log(X, y, weights):
    """! Confusion matrix

    Function to calculate confusion matrix, sensitivity and specificity for the logistic regression.

    @param X: Data
    @type X: Array

    @param y: Labels
    @type y: Array

    @param weights: Features
    @type weights: Array

    @return: Sensitivity and specificity
    @rtype: Floats

    """

    # Design matrix
    design_matrix = np.hstack((np.ones((X.shape[0], 1)), X))

    # Predictions
    predictions = sigmoid(np.dot(design_matrix, weights))

    # Convert probabilities to binary predictions
    binary_predictions = (predictions >= 0.5).astype(int)

    # Initialization of variables
    tp = 0              # True positive counts
    fp = 0              # False positive counts
    tn = 0              # True negative counts
    fn = 0              # False negative counts
    point_cluster1 = [] # Storage of cluster 1 points
    point_cluster2 = [] # Storage of cluster 2 points

    # Confusion matrix calculation and store points for the plot
    for pred, actual, point in zip(binary_predictions, y, X):
        # If it's a true positive
        if pred == 1 and actual == 1:
            tp += 1
            point_cluster1.append(point)
        # Elif it's a false positive
        elif pred == 1 and actual == 0:
            fp += 1
            point_cluster1.append(point)
        # Elif it's a true negative
        elif pred == 0 and actual == 0:
            tn += 1
            point_cluster2.append(point)
        # Elif it's a false negative
        elif pred == 0 and actual == 1:
            fn += 1
            point_cluster2.append(point)

    # Sensitivity and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Print results
    print("Confusion Matrix:\n")
    print("              Predict cluster 1   Predict cluster 2")
    print(f"Is cluster 1        {tn}                   {fp}")
    print(f"Is cluster 2        {fn}                   {tp}")
    print()
    print(f"Sensitivity (Successfully predict cluster 1): {sensitivity}")
    print(f"Specificity (Successfully predict cluster 2): {specificity}\n")
    # High sensitivity indicates the model's ability to correctly identify true positives (is the probability of a positive test result, conditioned on the individual truly being positive).
    # High specificity indicates its ability to correctly exclude false positives (is the probability of a negative test result, conditioned on the individual truly being negative).

    return point_cluster1, point_cluster2

def logistic_regression(X, y, epsilon):
    """! Logistic regression

    Function to make a Logistic regression.

    @param X: Data
    @type X: Array

    @param y: Labels
    @type y: Array
    
    """

    # Gradient Descent
    gradiant_weights = gradient_descent(X, y, learning_rate=0.01, max_iter=1000, epsilon=epsilon)

    # Result logistic regression for gradiant's method
    print("\nGradient descent:\n")
    print("w:")
    for weight in gradiant_weights:
        print(weight)
    print()

    # Confusion matrix and metrics for Gradient Descent
    gradiant_point_cluster1, gradiant_point_cluster2 = confusion_matrix_log(X, y, gradiant_weights)

    # Newton's Method
    newton_weights = newtons_method(X, y, max_iter=1000, epsilon=epsilon)

    # Result logistic regression for newton's method
    print("-------------------------------------------")
    print("Newton's method:\n")
    print("w:")
    for weight in newton_weights:
        print(weight)
    print()

    # Calculate confusion matrix and metrics for Newton method's
    newton_point_cluster1, newton_point_cluster2 = confusion_matrix_log(X, y, newton_weights)

    # Plot display
    plot_logistic_reg(X, y, gradiant_point_cluster1, gradiant_point_cluster2, newton_point_cluster1, newton_point_cluster2, gradiant_weights, newton_weights)

def plot_logistic_reg(X, y, gradiant_point_cluster1, gradiant_point_cluster2, newton_point_cluster1, newton_point_cluster2, gradiant_weights, newton_weights):
    """! Data plot

    Function to plot the data and compare ﻿different methods.

    @param X: Original data
    @type X: Array

    @param y: Labels
    @type y: Array

    @param gradiant_point_cluster1: Points in cluster 1 of the gradient descent method
    @type gradiant_point_cluster1: Array

    @param gradiant_point_cluster2: Points in cluster 2 of the gradient descent method
    @type gradiant_point_cluster2: Array

    @param newton_point_cluster1: Points in cluster 1 of the Newton's method
    @type newton_point_cluster1: Array

    @param newton_point_cluster2: Points in cluster 2 of the Newton's method
    @type newton_point_cluster2: Array

    @param gradiant_weights: Weights obtained from gradient descent
    @type gradiant_weights: Array

    @param newton_weights: Weights obtained from Newton's method
    @type newton_weights: Array

    """

    # Initialization of variables
    class1_data = X[y == 0]                                               # Cluster 1 on original data
    class2_data = X[y == 1]                                               # Cluster 2 on original data
    gradiant_x_cluster1 = [array[0] for array in gradiant_point_cluster1] # Abscissa on cluster 1 on gradient method
    gradiant_y_cluster1 = [array[1] for array in gradiant_point_cluster1] # Ordinate on cluster 1 on gradient method
    gradiant_x_cluster2 = [array[0] for array in gradiant_point_cluster2] # Abscissa on cluster 2 on gradient method
    gradiant_y_cluster2 = [array[1] for array in gradiant_point_cluster2] # Ordinate on cluster 2 on gradient method
    newton_x_cluster1 = [array[0] for array in newton_point_cluster1]     # Abscissa on cluster 1 on Newton's method
    newton_y_cluster1 = [array[1] for array in newton_point_cluster1]     # Ordinate on cluster 1 on Newton's method
    newton_x_cluster2 = [array[0] for array in newton_point_cluster2]     # Abscissa on cluster 2 on Newton's method
    newton_y_cluster2 = [array[1] for array in newton_point_cluster2]     # Ordinate on cluster 2 on Newton's method
    x_values = np.array([np.min(X[:, 0]), np.max(X[:, 0])])               # X values to the fitting line
    y_values_gradient = -(gradiant_weights[0] + gradiant_weights[1]*x_values) / gradiant_weights[2]
    y_values_newton = -(newton_weights[0] + newton_weights[1]*x_values) / newton_weights[2]
    
    # Plot display
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Ground Truth
    axs[0].scatter(class1_data[:, 0], class1_data[:, 1], c='r', marker='o', label='Cluster 1')
    axs[0].scatter(class2_data[:, 0], class2_data[:, 1], c='b', marker='o', label='Cluster 2')
    axs[0].set_title('Ground truth')
    axs[0].legend()

    # Gradient Descent
    axs[1].scatter(gradiant_x_cluster1, gradiant_y_cluster1, c='b', marker='o', label='Cluster 2')
    axs[1].scatter(gradiant_x_cluster2, gradiant_y_cluster2, c='r', marker='o', label='Cluster 1')
    axs[1].set_title('Gradient descent')
    axs[1].plot(x_values, y_values_gradient, c='black', label='Decision Boundary')
    axs[1].legend()

    # Newton's Method
    axs[2].scatter(newton_x_cluster1, newton_y_cluster1, c='b', marker='o', label='Cluster 2')
    axs[2].scatter(newton_x_cluster2, newton_y_cluster2, c='r', marker='o', label='Cluster 1')
    axs[2].set_title("Newton's method")
    axs[2].plot(x_values, y_values_newton, c='black', label="Decision Boundary")
    axs[2].legend()

    plt.tight_layout()
    plt.show()

def binarize_img(images):
    """! Image binarization

    Function for binarize images

    @param images: Images to binarize
    @type images: 2D Array(n, 28*28)

    @return: Images to binarize
    @rtype: 2D Array(n, 28*28)

    """

    # Initialization of variables
    binarized_img = np.zeros_like(images) # Empty matrix to store binarized images

    for i in range(len(images)):
        binarized_img[i] = np.where(images[i] > 0, 1, 0)
    
    return binarized_img

def E_step(binarized_images, class_probs, pixel_probs):
    """! Step E

    Function for the step E of the EM algorithm. This function calculates the average probability of each image belonging to each class.

    @param binarized_images: Binarized images
    @type binarized_images: 2D Array(n, 28*28)

    @param class_probs: Probability to belong to each class
    @type class_probs: Array(10)

    @param pixel_probs: Probability of each pixel being black
    @type pixel_probs: 2D Array(10, 28*28)

    @return: Mean for each image to belong to each class
    @rtype: 2D Array(n, 10)

    """

    # Initialization of variables
    num_img = binarized_images.shape[0]    # Number of images
    num_pixels = binarized_images.shape[1] # Number of pixels
    mean = np.zeros((num_img, 10))         # Empty matrix to store the mean

    # For each images
    for img in range(num_img):
        # For each cluster
        for cluster in range(10):
            mean[img][cluster] = class_probs[cluster]

            # For each pixel
            for pixel in range(num_pixels):
                # If it's a black pixel
                if binarized_images[img][pixel] == 1:
                    mean[img][cluster] *= pixel_probs[cluster][pixel]
                else:
                    mean[img][cluster] *= (1 - pixel_probs[cluster][pixel])
        
        # Normalizing the probabilities
        if sum(mean[img]) == 0:
            continue

        mean[img] /= sum(mean[img])
    
    return mean

def M_step(binarized_images, responsibility):
    """! Step M

    Function for the step m of the EM algorithm. This function update the parameters.

    @param binarized_images: Binarized images
    @type binarized_images: 2D Array(n, 28*28)

    @param responsibility: Weight matrix representing the responsibility of each image for each cluster
    @type responsibility: 2D Array(n, 10)
w
    @param p: Probability of each pixel being black
    @type p: 2D Array(10, 28*28)

    @return: Updated probabilities to belong to each class
    @rtype: Array(10,)

    @return: Updated probabilities of each pixel being black for each class
    @rtype: 2D Array(10, 28*28)

    """

    # Initialization of variables
    num_img = binarized_images.shape[0]      # Number of images
    num_pixels = binarized_images.shape[1]   # Number of pixels
    class_probs = np.zeros((10))             # Array to store updated class probabilities
    pixel_probs = np.zeros((10, num_pixels)) # Array to store updated pixel probabilities

    # For each class
    for cluster in range(10):
        # Update class probabilities
        class_probs[cluster] = sum(responsibility[:, cluster]) / num_img

        # For each pixel
        for pixel in range(num_pixels):
            cpt = 0

            # For each images
            for img in range(num_img):
                # If the pixel is black
                if binarized_images[img][pixel] == 1:
                    cpt +=  responsibility[img][cluster]
            if class_probs[cluster] == 0:
                continue

            # Update pixel probabilities
            pixel_probs[cluster][pixel] = cpt / (class_probs[cluster] * num_img)

    return class_probs, pixel_probs

def assign_label_and_test(w, labels, num_imgs):
    """! Labels assignation

    Function to associates labels with the data and evaluates the model's performance using the actual labels

    """

    mapping = np.zeros((10), dtype = int)
    counting = np.zeros((10, 10), dtype = int)
    prediction = np.zeros((num_imgs), dtype = int)

    # Cluster the images and count the relation between real label and each cluster
    for img in range(num_imgs):
        predict_cluster = np.argmax(w[img])
        counting[labels[img], predict_cluster] += 1
        prediction[img] = predict_cluster
    
    # Use array mapping to store the corresponding cluster of the specific label
    for _ in range(10):
        index = np.argmax(counting)
        real_label = index // 10
        cur_cluster = index % 10
        mapping[real_label] = cur_cluster
        counting[real_label, :] = 0
        counting[:, cur_cluster] = 0
    
    # Modify the prediction from cluster to real label
    mapping_inv = np.zeros((10), dtype = int)
    for i in range(10):
        mapping_cluster = mapping[i]
        mapping_inv[mapping_cluster] = i
    for img in range(num_imgs):
        predict_cluster = prediction[img]
        prediction[img] = mapping_inv[predict_cluster]

    return mapping, prediction

def print_imagination(p, mapping, num_row, num_col, labeled = False):
    """! Display imagination

    Function to print the imagined number.

    @param p: Probability of each pixel being black
    @type p: 2D Array(10, 28*28)

    @param mapping: Classes
    @type mapping: [10,]

    @param p: Probability of each pixel being black
    @type p: 2D Array(10, 28*28)

    @return: Updated probabilities to belong to each class
    @rtype: Array(10,)

    @return: Updated probabilities of each pixel being black for each class
    @rtype: 2D Array(10, 28*28)

    """
        
    # For each class
    for i in range(10):
        if labeled:
            print("Labeled", end=" ")
            
        print(f'Class {i}:')
        index = int(mapping[i])
        for row in range(num_row):
            for col in range(num_col):
                pixel = 1 if p[index][row * num_row + col] >= 0.5 else 0
                print(pixel, end = ' ')
            print('')
        print('')
    
    return

def confusion_matrix_em(y, prediction):
    """! Confusion matrix

    Function to calculate confusion matrix, sensitivity and specificity for the EM algorithm.

    @param y: True labels
    @type y: Array

    @param prediction: Predicted labels
    @type prediction: Array

    @param weights: Features
    @type weights: Array

    @return: Sensitivity and specificity
    @rtype: Floats

    """

    # Initialization of variables
    error = 0 # Error

    # For each class
    for j in range(10):
        # Initialization of variables
        confusion = np.zeros((2, 2), dtype = int) # Confusion matrix for a class

        for i in range(len(y)):
            if y[i] == j:
                # If it's a True Positive
                if prediction[i] == j:
                    confusion[0][0] += 1
                # Else if it's a False negative
                else:               
                    confusion[0][1] += 1
            else:
                # If it's a true Negative
                if prediction[i] != j:
                    confusion[1][1] += 1
                # Else if it's a False Positive
                else:
                    confusion[1][0] += 1

        # Sensitivity and specificity
        sensitivity = confusion[0][0] / (confusion[0][0] + confusion[0][1])
        specificity = confusion[1][1] / (confusion[1][0] + confusion[1][1])

        # Print results
        print("Confusion Matrix:\n")
        print(f"                   Predict number {j}   Predict not number {j}")
        print(f"Is number {j}                {confusion[0][0]}                   {confusion[0][1]}")
        print(f"Isn\'t number {j}             {confusion[1][0]}                   {confusion[1][1]}")
        print()
        print(f"Sensitivity (Successfully predict cluster i): {sensitivity}")
        print(f"Specificity (Successfully predict cluster i): {specificity}\n")

        error += confusion[0][0] 
    
    return 1 - error / len(y)

def em_algorithm(train_images, train_labels):
    # Initialization of variables 
    num_clusters = 10                                                  # Number of clusters
    max_iterations = 100                                               # Maximum iterations
    binarized_img = binarize_img(train_images)                         # Binarized images
    num_img = binarized_img.shape[0]                                   # Number of images
    num_pixels = binarized_img.shape[1]                                # Number of pixels
    class_probs = np.full((num_clusters), 0.1)                         # Probability to belong to a class
    pixel_probs = np.random.rand(num_clusters, num_pixels) / 2 + 0.25  # Chance of value 1 for number 0~9 and pixel 0~783, initially between 0.25~0.75
    prev_pixel_probs = np.zeros((num_clusters, num_pixels))            # Previous chance
    cluster_mapping = np.arange(num_clusters)                          # Mapping of differents clusters

    # For each iteration
    for i in range(max_iterations):
        # E step
        mean = E_step(binarized_img, class_probs, pixel_probs)

        # M step
        class_probs, pixel_probs = M_step(binarized_img, mean)

        # Print imagined number
        print_imagination(pixel_probs, cluster_mapping, 28, 28)

        # Difference between each iteration
        difference = sum(sum(abs(pixel_probs - prev_pixel_probs)))
        print(f'No. of Iteration: {i+1}, Difference: {difference}\n')
        print('------------------------------------------------------------')

        # Stop criteria : Choosing by experimenting different values
        if difference < 20 and len(binarized_img) <= 1000:
            break
        elif difference < 30 and len(binarized_img) > 1000:
            break

        # Previous chance update
        prev_pixel_probs = np.copy(pixel_probs)
    
    cluster_mapping, prediction = assign_label_and_test(mean, train_labels, num_img)
    print_imagination(pixel_probs, cluster_mapping, 28, 28, True)

    # Error calculation et confusion matrix
    error = confusion_matrix_em(train_labels, prediction)
    print(f'\nTotal iteration to converge: {i}')
    print(f'Total error rate: {error}')

if __name__ == "__main__":
    # Menu display
    print("\n################ MENU ################\n")
    print("1 - Logistic regression")
    print("2 - EM algorithm\n")
    print("######################################\n")

    choice = int(input('Which exercice ? '))
    switch(choice)  