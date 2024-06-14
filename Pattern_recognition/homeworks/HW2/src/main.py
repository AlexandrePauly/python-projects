import typing as t
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(self, inputs: npt.NDArray[float], targets: t.Sequence[int]) -> None:
        """! Logistic regression

        Function to make a Logistic regression.

        @param inputs: Data
        @type input: Array

        @param targets: Labels
        @type targets: Array
        
        """
        
        # Initialize weights and intercept
        self.weights = np.zeros(inputs.shape[1]) # Weights
        self.intercept = 0                       # Intercept
        n = len(targets)                         # Number of data
        
        # For each iterations
        for _ in range(self.num_iterations):
            # Predictions with the sigmoid function for the logistic regression
            predictions = self.sigmoid(np.dot(inputs, self.weights) + self.intercept)

            # Error calculation
            error = targets - predictions

            # Gradients calculation
            gradient_weights = np.dot(inputs.T, error) / n
            gradient_intercept = np.mean(error)

            # Weights and intercept update
            self.weights += self.learning_rate * gradient_weights
            self.intercept += self.learning_rate * gradient_intercept

    def predict(self, inputs: npt.NDArray[float]) -> t.Tuple[np.ndarray, np.ndarray]:
        """! Calulation of predicted value

        Function for calculate the prediction.

        @param inputs: Data
        @type input: Array

        @return: Predictions and classes predicted
        @rtype: Array and Array

        """
                
        # Predict class labels based on probability threshold 0.5
        predictions = self.sigmoid(np.dot(inputs, self.weights) + self.intercept)
        classes_predicted = np.where(predictions >= 0.5, 1, 0)

        return predictions, classes_predicted

    def sigmoid(self, x):
        """! Sigmoid function for logistic regression

        Function to Calculate the sigmoid value in x = np.dot(X,weights) + intercept.

        @param X: Value to predict
        @type X: Array

        @return: Predictions
        @rtype: Array
        
        """
            
        # Sigmoid function
        return 1 / (1 + np.exp(-x))


class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(self, inputs: npt.NDArray[float], targets: t.Sequence[int]) -> None:
        """! Fisher’s Linear Discriminant

        Function to make a FLD.

        @param inputs: Data
        @type input: Array

        @param targets: Labels
        @type targets: Array
        
        """
                
        # Class means calculation
        self.m0 = np.mean(inputs[targets == 0], axis=0)
        self.m1 = np.mean(inputs[targets == 1], axis=0)
        
        # Within-class scatter matrix calculation
        self.sw = np.cov(inputs.T)
        
        # Between-class scatter matrix calculation
        self.sb = np.outer((self.m1 - self.m0), (self.m1 - self.m0))
        
        # Fisher's linear discriminant calculation
        self.w = np.dot(np.linalg.inv(self.sw), (self.m1 - self.m0))

    def predict(self, inputs: npt.NDArray[float]) -> np.ndarray:
        """! Calulation of predicted value

        Function for calculate the prediction.

        @param inputs: Data
        @type input: Array

        @return: Predictions
        @rtype: Array

        """
                
        # Project inputs onto the discriminant axis
        projected = np.dot(inputs, self.w)
        threshold = (np.mean(np.dot(self.m0, self.w)) + np.mean(np.dot(self.m1, self.w))) / 2

        return (projected > threshold).astype(int)

    def plot_projection(self, inputs: npt.NDArray[float]):
        projections = np.dot(inputs, self.w)
        mean_projection = np.mean(projections)

        # Calculating the direction vector of the line
        direction_vector = np.array([1, self.w[1] / self.w[0]])

        plt.plot([0, direction_vector[0]], [0, direction_vector[1]], 'k', label='Projection Line')

        # Plotting data points with distances from the projection line
        for i, (projection, input_point) in enumerate(zip(projections, inputs)):
            projected_point = projection * direction_vector
            if projection >= mean_projection:
                plt.plot([projected_point[0], input_point[0]], [projected_point[1], input_point[1]], 'r', linewidth=0.5)  # Plotting lines from data points to projection line
                plt.scatter(projected_point[0], projected_point[1], c='r', marker='o')  # Plotting projection points
                plt.scatter(input_point[0], input_point[1], c='r', marker='o')  # Plotting points at the end of distance lines
            else:
                plt.plot([projected_point[0], input_point[0]], [projected_point[1], input_point[1]], 'b', linewidth=0.5)  # Plotting lines from data points to projection line
                plt.scatter(projected_point[0], projected_point[1], c='b', marker='o')  # Plotting projection points
                plt.scatter(input_point[0], input_point[1], c='b', marker='o')  # Plotting points at the end of distance lines

        plt.legend()
        plt.xlabel('Projection')
        plt.title('Projection of data points')
        plt.show()


def compute_auc(y_trues, y_preds) -> float:
    """! AUC calculation

    Function to Calculate the AUC indicator with the trapezoid rule.

    @param y_trues: Labels
    @type y_trues: Array

    @param y_preds: Predicted labels
    @type y_preds: Array

    @return: AUC
    @rtype: Float
    
    """
            
    # Sort predictions by decreasing order
    sorted_indices = sorted(range(len(y_preds)), key=lambda i: y_preds[i], reverse=True)
    y_trues_sorted = [y_trues[i] for i in sorted_indices]

    # Initialization of variables
    n = len(y_trues_sorted) # Number of data
    auc = 0.0               # AUC
    tp = 0                  # True Positive
    fp = 0                  # False Positive
    last_fpr = 0            # Last False Positive Rate

    # Calculate AUC using trapezoidal rule
    for i in range(n):
        if y_trues_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / sum(y_trues)       # True Positive Rate
        fpr = fp / (n - sum(y_trues)) # False Positive Rate
        auc += (fpr - last_fpr) * tpr # Area Under the Curve
        last_fpr = fpr                # Last False Positive Rate update

    return auc


def accuracy_score(y_trues, y_preds) -> float:
    """! Accuracy score

    Function to Calculate the accuracy score.

    @param y_trues: Labels
    @type y_trues: Array

    @param y_preds: Predicted labels
    @type y_preds: Array

    @return: accuracy score
    @rtype: Float
    
    """

    # Compute accuracy (TP + TN)
    correct = sum(1 for true, pred in zip(y_trues, y_preds) if np.all(true == pred))

    return correct / len(y_trues)


def main():
    # Initialization of variables
    train_df = pd.read_csv('../data/train.csv') # Training data
    test_df = pd.read_csv('../data/test.csv')   # Test data

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy() # Abscissa of training data (n_samples, n_features)
    y_train = train_df['target'].to_numpy()                # Ordinate of training data (n_samples, )
    x_test = test_df.drop(['target'], axis=1).to_numpy()   # Abscissa of test data (n_samples, n_features)
    y_test = test_df['target'].to_numpy()                  # Ordinate of test data (n_samples, )

    # Logistic regression method
    LR = LogisticRegression(learning_rate=1e-1, num_iterations=1000)
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy = {accuracy:.4f}, AUC = {auc_score:.4f}')

    # Part2: FLD
    cols = ['27', '30']  # Dont modify
    x_train = train_df[cols].to_numpy()     # Abscissa of training data
    y_train = train_df['target'].to_numpy() # Ordinate of training data
    x_test = test_df[cols].to_numpy()       # Abscissa of test data
    y_test = test_df['target'].to_numpy()   # Ordinate of teste data

    # FLD method
    FLD_ = FLD()
    FLD_.fit(x_train, y_train)
    y_preds = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_preds)
    logger.info(f'FLD: m0 = {FLD_.m0}, m1 = {FLD_.m1}')
    logger.info(f'FLD: \nSw = {FLD_.sw}')
    logger.info(f'FLD: \nSb = {FLD_.sb}')
    logger.info(f'FLD: \nw = {FLD_.w}')
    logger.info(f'FLD: Accuracy = {accuracy:.4f}')
    FLD_.plot_projection(x_test)


if __name__ == '__main__':
    main()