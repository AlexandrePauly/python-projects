#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import os
import numpy as np
import pylab
import generate_gif


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def sne(X=np.array([]), labels=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, method='t-sne'):
    """
        Runs SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
        Use two method : t-SNE or symmetric SNE.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Save figure
    save_figure(Y, labels, perplexity, method, False, 'epochs_0')

    # If we need to use t-sne method
    if method == 't-sne':  
        print(f'\nRunning t-SNE with perlexity = {perplexity} : \n')
    # Else, if we need to use symetric sne method
    elif method == "symmetric-sne":
        print(f'\nRunning symmetric SNE with perlexity = {perplexity} : \n') 

    # Run iterations
    for i in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)

        # If we need to use t-sne method
        if method == 't-sne':
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        # Else, if we need to use symetric sne method
        elif method == "symmetric-sne":
            num = np.exp(-np.add(np.add(num, sum_Y).T, sum_Y))
        else:
            print("Wrong method chosen !")
            break
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q

        # If we need to use t-sne method
        if method == 't-sne':
            for j in range(n):
                dY[j, :] = np.sum(np.tile(PQ[:, j] * num[:, j], (no_dims, 1)).T * (Y[j, :] - Y), 0)
        # Else, if we need to use symetric sne method
        elif method == "symmetric-sne":
            for j in range(n):
                dY[j, :] = np.sum(np.tile(PQ[:, j], (no_dims, 1)).T * (Y[j, :] - Y), 0)
            
        # Perform the update
        if i < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (i + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (i + 1, C))

        if (i + 1) % 50 == 0:
            # Save figure
            save_figure(Y, labels, perplexity, method, False, f'epochs_{i + 1}')

        # Stop lying about P-values
        if i == 100:
            P = P / 4.

    plot(P, Q, perplexity, method)

    # Return solution
    return Y

def save_figure(Y, labels, perplexity, method, init, name_file):
    """ Function to save figures. """
        
    if init:
        pylab.figure()
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)

    # If we need to use t-sne method
    if method == 't-sne':
        save_dir = f"sne_results/t-SNE/perplexity={int(perplexity)}"

        # Creating directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
        pylab.savefig(f'{save_dir}/{name_file}.png')
    # Else, if we need to use symetric sne method
    elif method == "symmetric-sne":
        
        save_dir = f"sne_results/symmetric_SNE/perplexity={int(perplexity)}"
        
        # Creating directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        pylab.savefig(f'{save_dir}/{name_file}.png')        
    pylab.cla()

def plot(P, Q, perplexity, method):
    """ Function to plot result """

    # If we need to use t-sne method
    if method == 't-sne':
        pylab.hist(P.flatten(),bins = 30,log = True)
        pylab.savefig(f'sne_results/t-SNE/perplexity={int(perplexity)}/P.png')
        pylab.cla()
        pylab.hist(Q.flatten(),bins = 30,log = True)
        pylab.savefig(f'sne_results/t-SNE/perplexity={int(perplexity)}/Q.png')
    # Else, if we need to use symetric sne method
    elif method == "symmetric-sne":
        pylab.hist(P.flatten(),bins = 30,log = True)
        pylab.savefig(f'sne_results/symmetric_SNE/perplexity={int(perplexity)}/P.png')
        pylab.cla()
        pylab.hist(Q.flatten(),bins = 30,log = True)
        pylab.savefig(f'sne_results/symmetric_SNE/perplexity={int(perplexity)}/Q.png')

if __name__ == "__main__":
    # Displaying messages
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")

    # Loading data
    X = np.loadtxt("../data/sne_python/mnist2500_X.txt")            # Inputs
    labels = np.loadtxt("../data/sne_python/mnist2500_labels.txt")  # Labels

    # Hyper-parameters
    method = ["t-sne", "symmetric-sne"]  # Method used (Values : t-sne or symmetric-sne)
    perplexity = [5, 15, 30, 50, 100]    # Perplexity values to test

    # For each perplexity
    for method_value in method:
        for perplexity_value in perplexity:
            # SNE method
            Y = sne(X, labels, 2, 50, perplexity_value, method_value)
            
            # If we need to use t-sne method
            if method_value == 't-sne':
                folder = f"sne_results/t-SNE/perplexity={int(perplexity_value)}"
            # Else, if we need to use symetric sne method
            elif method_value == "symmetric-sne":
                folder = f"sne_results/symmetric_SNE/perplexity={int(perplexity_value)}"
            
            # Generating gifs
            generate_gif.create_gifs(folder)