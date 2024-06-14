# Machine Learning - Homework 7

This project aims to implement face recognition using eigenface and fisherface on Yale face database and t-SNE on MNIST dataset.

This project was implemented using the Python language and the numpy, matplotlib.pyplot, Pillow and pylab libraries. The file sne.py is a modified file from https://lvdmaaten.github.io/tsne/

## Architecture
- data folder : Images/dataset (Faces and MNIST)
- doc folder : Requirements and report
- src : code and results for each clustering methods (a folder for each method and each image with GIF)

## Installation Instructions

1. Clone or download this repository to your local machine.

2. Ensure that you have an active internet connection to install external libraries.

3. Run the following commands if some of the libraries are not installed on your machine (prerequisite: Python) :

    ```pip install numpy```

    ```pip install matplotlib```

    ```pip install pillow```

    ```pip install pylab-sdk```

4. Go in the src directory and launch the application : ```python3 sne.py``` for SNE method and ```python3 kernel_eigenface.py``` for faces recognition methods.


>**_Warning :_** Avoid modifying the project's directory structure to prevent breaking URLs!

## Author

Created by Alexandre PAULY.