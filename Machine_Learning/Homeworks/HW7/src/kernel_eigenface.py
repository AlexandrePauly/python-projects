#!/usr/bin/env python3
"""! @brief Python program for Kernel eigenface."""
##
# @file kernel_eigenface.py
#
# @brief All homework 7 functions for eigenface.
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
import numpy as np
import os
from matplotlib import pyplot as plt

def loading_data(directory):
    """! Loading data

    Function for reading directory to load all images

    @param directory: File to read
    @type directory: String (file path)

    @return: Images, labels and filenames
    @rtype: Array, Array, Array

    """
        
    # Initialization of list to store data
    images = []     # List to store images
    labels = []     # List to store labels     
    filenames = []  # List to store filanames

    for filename in os.listdir(directory):
        filepath = f"{directory}/{filename}"

        # Load the PGM image
        with open(filepath, 'rb') as f:
            assert f.readline() == b'P5\n'
            f.readline()
            width, height = [int(i) for i in f.readline().split()]
            assert int(f.readline()) <= 255

            # Loops on each pixels
            img = np.zeros((height, width), dtype=np.uint8)
            for row in range(height):
                for col in range(width):
                    img[row][col] = ord(f.read(1))
            
            # Append image, label and filename
            images.append(img.flatten())
            file =' '.join(filename.split('.')[0:2])
            label = int(file[7:9])
            labels.append(label)
            filenames.append(file)
    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    filenames = np.array(filenames)
    
    return images, labels, filenames

def pre_processing(data):
    """! Data pre-processing 

    Function for resizing images.

    @param data: Images to pre-process
    @type data: Array

    @return: Compressed images
    @rtype: Array

    """
    
    # Initialization of variables
    num_imgs = data.shape[0]                                                     # Number of images
    original_height, original_width = 231, 195                                   # Original dimensions
    compressed_height, compressed_width = 77, 65                                 # New dimensions
    img_compressed = np.zeros((num_imgs, compressed_height * compressed_width))  # List of compressed images

    # For each images
    for img in range(num_imgs):
        # For each rows
        for row in range(compressed_height):
            # For each columns
            for col in range(compressed_width):
                tmp = 0

                for i in range(3):
                    for j in range(3):
                        tmp += data[img][(row*3 + i) * original_width + (col*3 + j)]

                img_compressed[img][row * compressed_width + col] = tmp // 9

    return img_compressed

def linear_kernel(u, v):
    """! Linear_kernel

    Function to calculate the linear kernel.

    @param u: Matrix 1
    @type u: 2D Array

    @param v: Matrix 2
    @type v: 2D Array

    @return: linear_kernel(u,v)
    @rtype: Array

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
    @rtype: Array

    """
    
    gamma = 1e-10
    dist = np.sum(u ** 2, axis=1).reshape(-1, 1) + np.sum(v ** 2, axis=1) - 2 * u @ v.T

    return np.exp(-gamma * dist)

def linear_and_RBFkernel(u, v):
    """! Linear + RBF kernel

    Function to calculate the addition of linear and RBF kernel.

    @param u: Matrix 1
    @type u: 2D Array

    @param v: Matrix 2
    @type v: 2D Array

    @return: linear_kernel(u,v) + RBFkernel(u,v)
    @rtype: Array

    """

    return linear_kernel(u, v) + RBFkernel(u, v)

def eigenvector_processing(eigenvalue, eigenvector):
    """! Eigen vector processing

    Function for processing eigenvector.

    @param eigenvalue: Eigenvalues
    @type eigenvalue: Array

    @param eigenvector: Eigenvectors
    @type eigenvector: Array

    @return: Eigen vectors
    @rtype: Array

    """
        
    # Sorting eigenvalues
    index = np.argsort(-eigenvalue)
    eigenvector = eigenvector[:, index]

    # Keep 25 first eigenvector
    W = eigenvector[:, :25]

    # Eigen vectors normalization
    for i in range(W.shape[1]):
        W[:, i] = W[:, i] / np.linalg.norm(W[:, i])

    return W

def PCA(data):
    """! PCA

    Function for compute PCA.

    @param data: Images for PCA
    @type data: Array

    @return: Eigen vectors
    @rtype: Array

    """
    
    # Computing covariance matrix
    covariance = np.cov(data.T)

    # Computing eigenvalue and eigenvector using covariance matrix
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    
    W = eigenvector_processing(eigenvalue, eigenvector)

    return W

def LDA(data, label):
    """! LDA

    Function for compute LDA.

    @param data: Images for LDA
    @type data: Array

    @return: Eigen vectors
    @rtype: Array

    """
        
    # Initialization of variables
    dimension = data.shape[1]              # Dimension of data
    Sw = np.zeros((dimension, dimension))  # Dispersion matrix intra-class
    Sb = np.zeros((dimension, dimension))  # Dispersoion matrix inter-class
    mean = np.mean(data, axis=0)           # Mean vector
    nb_label = np.max(label)

    # For each label
    for i in range(nb_label):
        # Subject to extract
        id = np.where(label == i + 1)
        subject = data[id]

        # Mean vector for this subject
        mean_subject = np.mean(subject, axis=0)

        # Difference within class
        within_diff = subject - mean_subject
        Sw += within_diff.T @ within_diff
        
        # Difference without class
        between_diff = mean_subject - mean
        Sb += 9 * between_diff.T @ between_diff

    # Compute transformation matrix
    Sw_Sb = np.linalg.pinv(Sw) @ Sb
    eigenvalue, eigenvector = np.linalg.eigh(Sw_Sb)

    W = eigenvector_processing(eigenvalue, eigenvector)

    return W

def PCA_kernel(data, kernel_type):
    """! kernel PCA

    Function for compute kernel PCA.

    @param data: Images for kernel PCA
    @type data: Array

    @param kernel_type: Kernel type
    @type kernel_type: Function

    @return: Eigen vectors
    @rtype: Array

    """

    # Centering data
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    
    # Computing kernel
    kernel = kernel_type(centered_data, centered_data)
    
    # Computing eigenvalue and eigenvector using covariance matrix
    eigenvalue, eigenvector = np.linalg.eigh(kernel)
    
    W = eigenvector_processing(eigenvalue, eigenvector)

    return W, kernel

def LDA_kernel(data, kernel_type):
    """! kernel LDA

    Function for compute kernel LDA.

    @param data: Images for kernel LDA
    @type data: Array

    @param kernel_type: Kernel type
    @type kernel_type: Function

    @return: Eigen vectors
    @rtype: Array

    """

    # Centering data
    mean = np.mean(data, axis=0)
    centered_data = data - mean
        
    Z = np.ones((centered_data.shape[0], centered_data.shape[0])) / 9

    # Computing kernel
    kernel = kernel_type(centered_data, centered_data)

    # Computing dispersion matrix
    Sw = kernel @ kernel      # Dispersion matrix intra-class
    Sb = kernel @ Z @ kernel  # Dispersion matrix inter-class
    
    # Compute transformation matrix
    Sw_Sb = np.linalg.pinv(Sw) @ Sb
    eigenvalue, eigenvector = np.linalg.eigh(Sw_Sb)
    
    W = eigenvector_processing(eigenvalue, eigenvector)

    return W, kernel

def plot(matrix, method):
    """! Plot faces function

    Function for plotting eigenfaces and fisherfaces

    @param matrix: Eigenvectors
    @type matrix: Array

    @param method: Method used
    @type method: String

    """
        
    fig = plt.figure(figsize=(5, 5))

    # For 25 first images
    for i in range(matrix.shape[1]):
        img = matrix[:, i].reshape(77, 65)
        ax = fig.add_subplot(5, 5, i + 1)
        ax.axis('off')
        ax.imshow(img, cmap='gray')
    
    # Saving plot
    fig.savefig(f'kernel_eigenfaces_results/{method}.jpg')

    plt.show()

def face_reconstruction(W, train_img, method):
    """! Plot faces function

    Function for reconstructing faces and saving results.

    @param W: Eigen vectors
    @type W: Array

    @param train_img: Training images
    @type train_img: Array

    @param method: Method used
    @type method: String

    """

    sample = np.random.choice(len(train_img), 10, replace=False)

    # Plot display
    fig = plt.figure(figsize=(8, 2))

    for i in range(10):
        img = train_img[sample[i]].reshape(77, 65)
        ax = fig.add_subplot(2, 10, i + 1)
        ax.axis('off')
        ax.imshow(img, cmap='gray')

        x = img.reshape(1, -1)

        # Face reconstruction
        reconstruct_img = x @ W @ W.T
        reconstruct_img = reconstruct_img.reshape(77, 65)
        ax = fig.add_subplot(2, 10, i + 11)
        ax.axis('off')
        ax.imshow(reconstruct_img, cmap='gray')

    # Saving reconstruction
    fig.savefig(f'kernel_eigenfaces_results/{method}_reconstruction.jpg')

    plt.show()

def prediction(train_img, train_label, test_img, test_label, W, output_name, kernel_bool=False, train_kernel=None, kernel_type=None):
    """ Preditions

    Function to predict faces for test images.

    @param train_img: Training images
    @type train_img: Array

    @param train_label: Labels for training images
    @type train_label: Array

    @param test_img: Test images
    @type test_img: Array

    @param test_label: Labels for test images
    @type test_label: Array

    @param W: Eigenvectors
    @type W: Array

    @param output_name: Name of output
    @type output_name: String

    @param kernel_bool: True or False if we need to use kernel or not
    @type kernel_bool: Boolean

    @param train_kernel: Precomputed kernel matrix for training data
    @type train_kernel: Array

    @param kernel_type: Type of kernel to use (linear_kernel or RBFkernel)
    @type kernel_type: String

    """
    
    # Initialization of variables
    errors = []

    for k in range(1, 30):
        error = 0   # Error

        if kernel_bool:
            # Centering data
            mean = np.mean(train_img, axis=0)
            centered_train = train_img - mean
            centered_test = test_img - mean

            # Computing kernel
            test_kernel = kernel_type(centered_test, centered_train)

            xW_train = train_kernel @ W
            xW_test = test_kernel @ W
        else:
            xW_train = train_img @ W
            xW_test = test_img @ W

        for i in range(len(test_img)):
            # Distance initialization
            distance = np.zeros(len(train_img))

            for j in range(len(train_img)):
                distance[j] = np.sum((xW_test[i] - xW_train[j]) ** 2)

            # Prediction by using the nearest neighbor
            neighbors = np.argsort(distance)[:k]
            prediction = np.argmax(np.bincount(train_label[neighbors]))

            if test_label[i] != prediction:
                error += 1

        errors.append(error / 30 * 100)

    # Plotting errors
    plot_errors(errors, output_name)
    
    infos_error(errors)

def infos_error(errors):
    """ Infos error

    Function to display infos on errors (mean, min, max).

    @param errors: List of error rates for different values of k
    @type errors: List

    """
        
    # Initialization of variables
    mean = np.mean(errors)  # Mean error
    min_error = 100         # Min error
    min_index = -1          # Min index
    max_error = 0           # Max error
    max_index = -1          # Max index

    # For each error
    for i in range(len(errors)):
        if errors[i] > max_error:
            max_error = errors[i]
            max_index = i + 1
        
        if errors[i] < min_error:
            min_error = errors[i]
            min_index = i + 1

    # Messages display
    print("\nMean error = ", mean)
    print(f"Min error = {min_error} with k = {min_index}")
    print(f"Max error = {max_error} with k = {max_index}")

def plot_errors(errors, method):
    """ Plotting error

    Function to plot error rates against the value of k.

    @param errors: List of error rates for different values of k
    @type errors: List

    @param method: Method used
    @type method: String

    """

    k_values = range(1, len(errors) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, errors, marker='o', linestyle='-', color='b')
    plt.title('Error Rate vs. k Value')
    plt.xlabel('k Value')
    plt.ylabel('Error Rate (%)')
    plt.grid(True)
    plt.savefig(f'kernel_eigenfaces_results/{method}_errors_prediction.jpg')
    plt.show()

if __name__ == "__main__":
    # Menu display
    print("\n################ MENU ################\n")
    print("1 - PCA")
    print("2 - LDA")
    print("3 - PCA using RBF kernel")
    print("4 - PCA using linear kernel")
    print("5 - PCA using linear and RBF kernel")
    print("6 - LDA using RBF kernel")
    print("7 - LDA using linear kernel")
    print("8 - LDA using linear and RBF kernel\n")
    print("######################################\n")

    method = int(input('Choose your method : '))

    # Path to directory
    train_dir = '../data/Yale_Face_Database/Training'  # Training dataset
    test_dir = '../data/Yale_Face_Database/Testing'    # Testing dataset

    # Loading images
    train_img, train_label, train_filename = loading_data(train_dir)  # Training dataset
    test_img, test_label, test_filename = loading_data(test_dir)      # Testing dataset

    # Pre-processing images
    train_img = pre_processing(train_img)  # Training dataset
    test_img = pre_processing(test_img)    # Testing dataset
    
    # Method used for a given value
    if method == 1:
        W = PCA(train_img)
        plot(W, 'pca')
        face_reconstruction(W, train_img, 'pca')
        prediction(train_img, train_label, test_img, test_label, W, output_name='pca')
    elif method == 2:
        W = LDA(train_img, train_label)
        plot(W, 'lda')
        face_reconstruction(W, train_img, 'lda')
        prediction(train_img, train_label, test_img, test_label, W, output_name='lda')
    elif method == 3:
        W, train_kernel = PCA_kernel(train_img, RBFkernel)
        prediction(train_img, train_label, test_img, test_label, W, output_name='pca_RBFkernel', kernel_bool=True, train_kernel=train_kernel, kernel_type=RBFkernel)
    elif method == 4:
        W, train_kernel = PCA_kernel(train_img, linear_kernel)
        prediction(train_img, train_label, test_img, test_label, W, output_name='pca_linear_kernel', kernel_bool=True, train_kernel=train_kernel, kernel_type=linear_kernel)
    elif method == 5:
        W, train_kernel = PCA_kernel(train_img, linear_and_RBFkernel)
        prediction(train_img, train_label, test_img, test_label, W, output_name='pca_linear_and_RBF_kernel', kernel_bool=True, train_kernel=train_kernel, kernel_type=linear_and_RBFkernel)
    elif method == 6:
        W, train_kernel = LDA_kernel(train_img, RBFkernel)
        prediction(train_img, train_label, test_img, test_label, W, output_name='lda_RBFkernel', kernel_bool=True, train_kernel=train_kernel, kernel_type=RBFkernel)
    elif method == 7:
        W, train_kernel = LDA_kernel(train_img, linear_kernel)
        prediction(train_img, train_label, test_img, test_label, W, output_name='lda_linear_kernel', kernel_bool=True, train_kernel=train_kernel, kernel_type=linear_kernel)
    elif method == 8:
        W, train_kernel = LDA_kernel(train_img, linear_and_RBFkernel)
        prediction(train_img, train_label, test_img, test_label, W, output_name='lda_linear_and_RBF_kernel', kernel_bool=True, train_kernel=train_kernel, kernel_type=linear_and_RBFkernel)
    else:
        print("No method chosen !")