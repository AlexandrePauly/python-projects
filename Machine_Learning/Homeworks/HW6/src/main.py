#!/usr/bin/env python3
"""! @brief Python program for clustering methods."""
##
# @file main.py
#
# @brief All homework 6 functions.
#
##
#
# @section Libraries/Modules
# - matplotlib.pyplot extern library (https://matplotlib.org/stable/)
# - numpy extern library (https://numpy.org/)
# - os intern library (https://docs.python.org/3/library/os.html)
# - PIL extern library (https://pypi.org/project/pillow/)
# - scipy extern library (https://scipy.org/)
#
# @section Auteur
# - PAULY Alexandre
##

# Imported library
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from scipy.spatial.distance import cdist

def switch(exercise):
    """! Menu function

    Call function according to choice.

    @param exercice: Selected exercice
    @type exercice: Integer

    """

    # Initialization of variables
    img_size = 10000                                                  # Size of images
    gamma_s = 0.0001                                                  # Sensitivity to color differences between pixels
    gamma_c = 0.0001                                                  # Sensitivity to spatial distances between pixels
    num_cluster = 4                                                   # Number of clusters
    mode = 'ratio'                                                    # Mode used (ratio or normalized)
    method = 'random'                                                 # Method used (random or k-means)
    image_path = f'../data/image2.png'                                # Path to images

    # Loading data
    img = np.asarray(Image.open(image_path).getdata())

    # Compute the affinity matrix
    W = affinity_matrix(img, img_size, gamma_s, gamma_c)

    if exercise == 1:
        # Initialization of variables
        converge = 0                                             # Boolean (0: False and 1: True)
        iteration = 1                                            # Iterations counter
        output_dir = f'./results/kernel_kmeans/{method}/image2'  # Folder to save results

        # Initialize clusters and cluster counts
        clusters, C = init(W, method, img_size, num_cluster, output_dir)

        while not converge:
            print(f'iteration: {iteration}')
            pre_clusters = clusters

            clusters, C = kernel_kmeans(W, clusters, C, img_size, num_cluster)

            # Save image for this iteration
            save_picture(clusters, iteration, num_cluster, img_size, output_dir)

            converge = check_converge(clusters, pre_clusters, img_size)
            
            iteration += 1
    elif exercise == 2 :
        output_dir = f'./results/spectral_clustering/{mode}/{method}/image2'  # Folder to save results

        # Compute the Laplacian and degree matrix
        L, D = compute_laplacian(W)

        # Perform spectral clustering based on the chosen mode
        if mode == 'ratio':
            # Compute the eigen-decomposition of the Laplacian matrix
            U = eigen_decomposition(L, num_cluster)

            # Perform K-means clustering on the eigenvectors
            clusters = kmeans(U, num_cluster, output_dir, img_size, method)

            # If the number of clusters is less than or equal to 3, draw the eigenspace
            if num_cluster <= 3:
                draw_eigenspace(U, clusters, num_cluster, img_size, output_dir)

        elif mode == 'normalized':
            # Compute the normalized Laplacian matrix and its sqrt of the degree matrix
            L_normal, sqrt_D = normalize_laplacian(L, D)

            # Compute the eigen-decomposition of the normalized Laplacian matrix
            U = eigen_decomposition(L_normal, num_cluster)

            # Transform the eigenvectors using the sqrt of the degree matrix
            T = sqrt_D @ U

            # Perform K-means clustering on the transformed eigenvectors
            clusters = kmeans(T, num_cluster, output_dir, img_size, method)

            # If the number of clusters is less than or equal to 3, draw the eigenspace
            if num_cluster <= 3:
                draw_eigenspace(T, clusters, num_cluster, img_size, output_dir)
    else:
        print("\nERROR : You have made no choice")

def create_gifs(folder):
    # Searching png
    for root, dirs, files in os.walk(folder):
        # Storing paths
        images_by_cluster = {}
        
        for filename in files:
            if filename.endswith('.png') and "eigenspace" not in filename:
                # Case extraction (grouped by number of cluster)
                group = filename.split('_')[0]

                if group not in images_by_cluster:
                    images_by_cluster[group] = []

                images_by_cluster[group].append(os.path.join(root, filename))

        # Create a gif
        for cluster, image_paths in images_by_cluster.items():
            # Sorting images by path
            image_paths.sort()

            # Load images
            images = [Image.open(image_path) for image_path in image_paths]

            # Creating GIF
            gif_path = os.path.join(root, f'cluster={cluster}.gif')

            # Saving GIF
            images[0].save(gif_path, save_all=True, append_images=images[1:], duration=200, loop=0)

            print(f'GIF created for group {cluster} in {root}: {gif_path}')

def init_cluster_c(img_kernel, centers, img_size, num_cluster, output_dir):
    """! Cluster cournt array

    Function to initialize clusters based on initial centers.

    @param img_kernel: Affinity matrix.
    @type img_kernel: Array

    @param centers: Cluster centers
    @type centers: Integer

    @param img_size: Size of the images
    @type img_size: Integer

    @param num_cluster: Number of clusters.
    @type num_cluster: Integer

    @param output_dir: Directory to save results.
    @type output_dir: String

    @return: Initial cluster assignments and initial cluster counts.
    @rtype: Array and Integer

    """
    
    # Initialize cluster assignments
    clusters = np.zeros(img_size, dtype=int)

    # For each pixel
    for pixel in range(img_size):

        # Skip if the pixel is an initial center
        if pixel in centers:
            continue

        # Initialize minimum distance to infinity
        min_dist = np.Inf

        # For each cluster
        for cluster in range(num_cluster):

            # Get the current cluster center
            center = centers[cluster]
            temp_dist = img_kernel[pixel][center]

            # Update minimum distance and assign pixel to the nearest cluster
            if temp_dist < min_dist:
                min_dist = temp_dist
                clusters[pixel] = cluster

    # Construct cluster counts
    C = np.bincount(clusters, minlength=num_cluster)

    # Save initial clustering result
    save_picture(clusters, 0, num_cluster, img_size, output_dir)

    return clusters, C

def init(img_kernel, method, img_size, num_cluster, output_dir):
    """! Cluster cournt array

    Function to initialize clusters based on initial centers.

    @param img_kernel: Affinity matrix.
    @type img_kernel: Array

    @param method: Method (random or k-means)
    @type method: String

    @param img_size: Size of the images
    @type img_size: Integer

    @param num_cluster: Number of clusters.
    @type num_cluster: Integer

    @param output_dir: Directory to save results.
    @type output_dir: String

    @return: Initial cluster assignments and initial cluster counts.
    @rtype: Array and Integer

    """
        
    if method == 'random':
        centers = np.random.randint(img_size, size=num_cluster)
        clusters, C = init_cluster_c(img_kernel, centers, img_size, num_cluster, output_dir)
    elif method == 'k-means':
        centers = np.zeros(num_cluster, dtype=int)
        centers[0] = np.random.randint(img_size, size=1)

        # For each cluster
        for i in range(1, num_cluster):
            distances = np.zeros(img_size)

            # For each pixel
            for pixel in range(img_size):
                min_dist = np.Inf

                for k in range(i):
                    temp_dist = img_kernel[centers[k]][pixel]

                    # Update minimum distance
                    if temp_dist < min_dist:
                        min_dist = temp_dist

                distances[pixel] = min_dist

            # Normalize distances
            distances = distances / np.sum(distances)

            # Select next center based on distances
            centers[i] = np.random.choice(10000, size=1, p=distances)
        clusters, C = init_cluster_c(img_kernel, centers, img_size, num_cluster, output_dir)
    else:
        print("No mode chosen!")

    return clusters, C

def sigma_n(pixel_kernel, clusters, k, C):
    """! Compute the sum of pixel_kernel values for pixels in cluster k

    @param pixel_kernel: Kernel values for each pixel.
    @type pixel_kernel: Array

    @param clusters: Cluster assignments for each pixel.
    @type clusters: Array

    @param k: Current cluster index.
    @type k: Integer

    @param C: Array of cluster counts.
    @type C: Array

    @return: Scaled sum of pixel_kernel values for cluster k.
    @rtype: Float
    
    """

    # Use a boolean mask to select elements of pixel_kernel belonging to cluster k
    mask = clusters == k
    sum = np.sum(pixel_kernel[mask])
    
    return 2 / C[k] * sum

def sigma_pq(img_kernel, clusters, C, num_cluster):
    """! Compute the sum of img_kernel values for each cluster, excluding inter-cluster values

    @param img_kernel: Kernel values for each pixel pair.
    @type img_kernel: Array

    @param clusters: Cluster assignments for each pixel.
    @type clusters: Array

    @param C: Array of cluster counts.
    @type C: Array

    @param num_cluster: Number of clusters.
    @type num_cluster: Integer

    @return: Array of scaled sums of img_kernel values for each cluster.
    @rtype: Array

    """
    
    sum = np.zeros(num_cluster)

    # For each cluster
    for k in range(num_cluster):
        # Create a mask for pixels belonging to cluster k
        mask = clusters == k

        # Extract the submatrix of img_kernel corresponding to cluster k
        submatrix = img_kernel[mask][:, mask]

        # Sum the submatrix and scale it by the cluster size squared
        sum[k] = np.sum(submatrix) / (C[k] ** 2)

    return sum

def kernel_kmeans(img_kernel, clusters, C, img_size, num_cluster):
    """! Perform Kernel K-means clustering

    @param img_kernel: Kernel values for each pixel pair.
    @type img_kernel: Array

    @param clusters: Cluster assignments for each pixel.
    @type clusters: Array

    @param C: Array of cluster counts.
    @type C: Array

    @param img_size: Size of the images
    @type img_size: Integer

    @param num_cluster: Number of clusters.
    @type num_cluster: Integer

    @return: Updated cluster assignments and cluster counts.
    @rtype: Tuple (Array, Array)
    
    """
    
    # Initialization of variables
    new_clusters = np.zeros(img_size, dtype=int)         # New cluster
    pq = sigma_pq(img_kernel, clusters, C, num_cluster)  # 

    # For each pixel
    for pixel in range(img_size):
        distances = np.zeros(num_cluster)

        # For each cluster
        for cluster in range(num_cluster):
            distances[cluster] = img_kernel[pixel][pixel]
            distances[cluster] -= sigma_n(img_kernel[pixel, :], clusters, cluster, C)
            distances[cluster] += pq[cluster]

        new_clusters[pixel] = np.argmin(distances)

    # New cluster creation
    new_C = np.bincount(new_clusters, minlength=num_cluster)
        
    return new_clusters, new_C

def affinity_matrix(img, img_size, gamma_s, gamma_c):
    """! Compute affinity matrix

    Function to compute the affinity matrix using a combined spatial and color Gaussian kernel.

    @param img: Images
    @type img: Array

    @param img_size: Size of the images
    @type img_size: Integer

    @param gamma_s: Sensitivity to color differences between pixels
    @type gamma_s: Float

    @param gamma_c: Sensitivity to spatial distances between pixels
    @type gamma_c: Float

    """
        
    # Initialization of variables
    color_dist = cdist(img, img, 'sqeuclidean')  # Color distance between pixels
    coordinates = np.zeros((img_size, 2))        # Coordinates matrix for spatial distances

    # Creating a coordinate matrix for spatial distances
    for i in range(100):
        index = i * 100
        for j in range(100):
            coordinates[index + j][0] = i
            coordinates[index + j][1] = j

    # Calculate spatial distances between pixels
    spatial_distance = cdist(coordinates, coordinates, 'sqeuclidean')

    # Combine color and spatial distances with Gaussian kernels
    img_kernel = np.multiply(np.exp(-gamma_c * color_dist), np.exp(-gamma_s * spatial_distance))

    return img_kernel

def compute_laplacian(W):
    """! Compute Laplacian

    Function to compute the Laplacian matrix.

    @param W: Affinity matrix
    @type W: Array

    @return: Unnormalized Laplacian matrix and degree matrix
    @rtype: Matrix and Matrix

    """
        
    # Compute the degree matrix
    D = np.diag(np.sum(W, axis=1))

    # Compute the unnormalized Laplacian matrix
    L = D - W

    return L, D

def normalize_laplacian(L, D):
    """! Normalize Laplacian

    Function to normalize the Laplacian matrix.

    @param L: Laplacian matrix
    @type L: Matrix

    @param D: Degree matrix
    @type D: Matrix

    @return: Normalized Laplacian matrix and normalized degree matrix
    @rtype: Matrix and Matrix

    """
        
    # Compute the inverse square root of the degree matrix
    sqrt_D_inv = np.diag(1.0 / np.sqrt(np.diag(D)))

    # Compute the normalized Laplacian matrix
    L_normalized = sqrt_D_inv @ L @ sqrt_D_inv

    return L_normalized, sqrt_D_inv

def eigen_decomposition(L, num_cluster):
    """! Eigen decomposition

    Function to perform eigen decomposition and select the first k eigenvectors.

    @param L: Laplacian matrix
    @type L: Matrix

    @param num_cluster: Number of clusters
    @type num_cluster: Integer

    @return: First k eigenvectors
    @rtype: Vector

    """
        
    eigenvalues, eigenvectors = np.linalg.eig(L)
    index = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, index]

    return eigenvectors[:, 1:1 + num_cluster].real

def init_means_clusters(U, centers, num_cluster, img_size):
    """! Means clusters

    Function to initialize means and clusters for K-means.

    @param U: Laplacian matrix
    @type U: Matrix

    @param num_cluster: Number of clusters
    @type num_cluster: Integer

    @param img_size: Size of the images
    @type img_size: Integer

    @return: Means ans clusters
    @rtype: Array and Array

    """
        
    # Initialization of variables    
    means = np.zeros((num_cluster, num_cluster))  # Means of clusters
    clusters = np.full(img_size, -1, dtype=int)   # Clusters

    # Initializing clusters
    for i in range(num_cluster):
        means[i] = U[centers[i]]
        clusters[centers[i]] = i

    return means, clusters

def squared_distance(center, cur_point, num_cluster):
    """! Means clusters

    Function to calculate squared distance between a center and a current point.

    @param center: Laplacian matrix
    @type center: Matrix

    @param cur_point: Number of clusters
    @type cur_point: Integer

    @param num_cluster: Number of clusters
    @type num_cluster: Integer

    @return: Squared distance
    @rtype: Float

    """
    
    # Counter for the distance
    distance = 0

    # For each cluster
    for i in range(num_cluster):
        distance += (center[i] - cur_point[i]) ** 2
    
    return distance

def init_kmeans(U, num_cluster, method, img_size):
    """ Initialize K-means clustering.

    This function initializes the cluster centers for the K-means algorithm.

    @param U: Laplacian matrix
    @type U: Matrix

    @param num_cluster: Number of clusters
    @type num_cluster: Integer

    @param method: Method to initialize clusters ('random' or 'k-means')
    @type method: String

    @param img_size: Size of the images
    @type img_size: Integer

    @return: Initial cluster means and assignments
    @rtype: Tuple of arrays
    
    """

    if method == 'random':
        # Randomly select initial cluster centers
        centers = np.random.randint(img_size, size=num_cluster)
        means, clusters = init_means_clusters(U, centers, num_cluster, img_size)
    elif method == 'k-means':
        # Initialize the first center randomly
        centers = np.zeros(num_cluster, dtype=int)
        centers[0] = np.random.randint(img_size, size=1)

        # Use k-means to select subsequent centers
        for i in range(1, num_cluster):
            distances = np.zeros(img_size)

            for j in range(img_size):
                min_dist = np.Inf

                for k in range(i):
                    # Calculate the squared distance to the nearest center
                    temp_dist = squared_distance(U[centers[k]], U[j], num_cluster)

                    if temp_dist < min_dist:
                        min_dist = temp_dist

                distances[j] = min_dist

            # Normalize distances to form a probability distribution
            distances = distances / np.sum(distances)

            # Choose a new center based on the probability distribution
            centers[i] = np.random.choice(10000, size=1, p=distances)

        means, clusters = init_means_clusters(U, centers, num_cluster, img_size)

    return means, clusters

def E_step(U, means):
    """! E-step

    Function to perform the E-step of the K-means algorithm.
    
    @param U: Laplacian matrix
    @type U: Matrix

    @param means: Means of the clusters
    @type means: Array

    @return: New clusters
    @rtype: Array

    """
    
    # Compute the squared distances between each point and each mean
    distances = np.linalg.norm(U[:, np.newaxis] - means, axis=2)**2

    # Assign each point to the cluster with the minimum distance
    new_clusters = np.argmin(distances, axis=1)

    return new_clusters

def M_step(U, clusters, num_cluster):
    """ M-step

    Function to perform the M-step of the K-means algorithm.
    
    @param U: Laplacian matrix
    @type U: Matrix

    @param clusters: Current cluster assignments
    @type clusters: Array

    @param num_cluster: Number of clusters
    @type num_cluster: Integer

    @return: Updated means of the clusters
    @rtype: Array
    
    """
    
    # Initialize an array to store the new means of the clusters
    new_means = np.zeros((num_cluster, U.shape[1]))

    # For each cluster, compute the mean of the points assigned to that cluster
    for cluster in range(num_cluster):
        cluster_points = U[clusters == cluster]
        if len(cluster_points) > 0:
            new_means[cluster] = cluster_points.mean(axis=0)

    return new_means

def check_converge(clusters, pre_clusters, img_size):
    """ Check convergence of K-means.

    This function checks if the cluster assignments have converged.

    @param clusters: Current cluster assignments
    @type clusters: Array

    @param pre_clusters: Previous cluster assignments
    @type pre_clusters: Array

    @param img_size: Size of the images
    @type img_size: Integer

    @return: Convergence status (1 if converged, 0 otherwise)
    @rtype: Integer

    """
    
    for pixel in range(img_size):
        if clusters[pixel] != pre_clusters[pixel]:
            return 0  # Not converged
    return 1  # Converged

def save_picture(clusters, iteration, num_cluster, img_size, output_dir):
    """ Save the clustering result as an image.

    This function saves the current state of the clustering as an image file.

    @param clusters: Current cluster assignments
    @type clusters: Array

    @param iteration: Current iteration number
    @type iteration: Integer

    @param num_cluster: Number of clusters
    @type num_cluster: Integer

    @param img_size: Size of the images
    @type img_size: Integer

    @param output_dir: Directory to save the image
    @type output_dir: String
    
    """

    # Initialization of variables
    color = np.array([[56, 207, 0], [64, 70, 230], [186, 7, 61], [245, 179, 66], [55, 240, 240]])  # Define colors for the clusters
    pixel = np.zeros((10000, 3))                                                                   # Array to store the pixel colors
    
    for i in range(img_size):
        pixel[i, :] = color[clusters[i], :]

    # Reshape the pixel array to form an image
    pixel = np.reshape(pixel, (100, 100, 3))

    # Create an image from the pixel data
    img = Image.fromarray(np.uint8(pixel))

    # Save the image
    img.save(output_dir + '/%01d_%03d.png' % (num_cluster, iteration), 'png')

def kmeans(U, num_cluster, output_dir, img_size, method):
    """ Perform K-means clustering.

    This function performs K-means clustering on the data.

    @param U: Laplacian matrix or transformed matrix
    @type U: Matrix

    @param num_cluster: Number of clusters
    @type num_cluster: Integer

    @param output_dir: Directory to save the results
    @type output_dir: String

    @param img_size: Size of the images
    @type img_size: Integer

    @return: Final cluster assignments
    @rtype: Array
    
    """

    # Initialization of variables
    converge = 0                                                     # Convergence flag
    iteration = 1                                                    # Iteration counter
    means, clusters = init_kmeans(U, num_cluster, method, img_size)  # Initialize clusters

    while not converge:
        print(f'iteration: {iteration}')
        pre_clusters = clusters

        # E-step
        clusters = E_step(U, means)

        # M-step
        means = M_step(U, clusters, num_cluster)

        # Save the current clustering result
        save_picture(clusters, iteration, num_cluster, img_size, output_dir)

        # Check for convergence
        converge = check_converge(clusters, pre_clusters, img_size)
    
        iteration += 1

    return clusters


def draw_eigenspace(U, clusters, num_cluster, img_size, output_dir):
    """ Draw eigenspace of the clustering result.

    This function visualizes the clustering result in eigenspace.

    @param U: Laplacian matrix or transformed matrix
    @type U: Matrix

    @param clusters: Final cluster assignments
    @type clusters: Array

    @param num_cluster: Number of clusters
    @type num_cluster: Integer

    @param img_size: Size of the images
    @type img_size: Integer

    @param output_dir: Directory to save the plot
    @type output_dir: String
    
    """

    # Initialization of variables
    points_x, points_y, points_z = [], [], []  # Store coordinates
    color = ['c', 'm', 'grey']                 # Colors
    
    if num_cluster == 2:
        # Prepare lists for 2D coordinates
        for _ in range(num_cluster):
            points_x.append([])
            points_y.append([])

        # Assign points to clusters
        for pixel in range(img_size):
            points_x[clusters[pixel]].append(U[pixel][0])
            points_y[clusters[pixel]].append(U[pixel][1])
        
        # Plot the clusters
        for cluster in range(num_cluster):
            plt.scatter(points_x[cluster], points_y[cluster], c=color[cluster])
        
        # Save the plot
        plt.savefig(f'{output_dir}/eigenspace_{num_cluster}.png')
    elif num_cluster == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # Prepare lists for 3D coordinates
        for _ in range(num_cluster):
            points_x.append([])
            points_y.append([])
            points_z.append([])

        # Assign points to clusters
        for pixel in range(img_size):
            points_x[clusters[pixel]].append(U[pixel][0])
            points_y[clusters[pixel]].append(U[pixel][1])
            points_z[clusters[pixel]].append(U[pixel][2])

        # Plot the clusters in 3D
        for cluster in range(num_cluster):
            ax.scatter(points_x[cluster], points_y[cluster], points_z[cluster], c=color[cluster])

        # Save the plot
        fig.savefig(f'{output_dir}/eigenspace_{num_cluster}.png')
    plt.show()

if __name__ == "__main__":
    # Menu display
    print("\n################ MENU ################\n")
    print("1 - Kernel k-means")
    print("2 - Spectral clustering\n")
    print("######################################\n")

    choice = int(input('Which exercice ? '))
    switch(choice)  

    # Generate gifs
    # folder = './results'
    # create_gifs(folder)