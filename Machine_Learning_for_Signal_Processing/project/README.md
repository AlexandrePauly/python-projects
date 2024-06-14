# Recognition in natural environments II : Image Data Generation

This project aims to detect road contours in images using a neural network model and post-processing technics. The detected contours are used to materialise the edges of rivers or roads for an antonomous navigation.

This project was implemented using the Python language and the numpy, opencv, tensorflow, matplotlib.pyplot and sklearn libraries.

## Description

The project consists of the following elements

1. **data**: A dataset of annotated road images with their corresponding contours. This data is used to train model.

2. **Models**: A pre-trained U-Net and V-Net model for edge detection. Models are trained on the training data and used to predict edges in new images.

3. **Python Scripts**: Python scripts to load data, train the model, make predictions on new images and evaluate model performance.

## Project Structure
- data/ : All training and test dataset (images and labels) in zip file
- doc/ : Requirements and report files
- src/ : 
    - models/ : Pre-trained models
    - predictions/ : Best predictions
    - ntebook : code

## Installation Instructions

1. Clone or download this repository to your local machine.

2. Ensure that you have an active internet connection to install external libraries.

3. Run the following commands if some of the libraries are not installed on your machine (prerequisite: Python) :

    ```pip install numpy```

    ```pip install opencv-python```

    ```pip install tensorflow```

    ```pip install matplotlib```

    ```pip install scikit-learn```

4. Run all functions.

## Instructions for Use

1. **Install Dependencies** :
   - Make sure you have installed all the necessary dependencies listed in the `installations instructions` section.

2. **Model Training**:
   - Train the model in the `Train model` section of `main.ipynb` file or use a pre-trained model.

3. **Predictions**:
   - Predict in the `Predict the model` section of `main.ipynb` file.

4. **Evaluation**:
   - Evaluate the performance of the model by comparing the predictions with the actual annotations.

## Authors

- Maxime Clémeanceau : maxime.clemenceau@cy-tech.fr
- Alexandre PAULY : alexandre.pauly@cy-tech.fr