import numpy as np
import pandas as pd
from loguru import logger


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        """! Linear regression method

        Function for regression using the close form.

        @param x: Abscissa data
        @type x: Array

        @param y: Ordinate data
        @type y: Array

        """

        # Calculation of Features
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

        # Calculation of intercept
        self.intercept = np.mean(y) - np.mean(X, axis=0).dot(self.weights)

    def predict(self, X):
        """! Calulation of predicted value

        Function for calculate the prediction.

        @param X: Abscissa data
        @type X: Array

        @return: Predictions
        @rtype: Float

        """

        # Prediction calculating with weight 
        return X @ self.weights + self.intercept
    

class LinearRegressionGradientdescent(LinearRegressionBase):
    def fit(self, X, y, learning_rate: float = 0.001, epochs: int = 1000):
        """! Gradient descent method 

        Function for regression using the gradient descente.

        @param X: Abscissa data
        @type X: Array

        @param X: Ordinate data
        @type X: Array

        @return: Predictions
        @rtype: Float

        """
        
        # Initialization of variables
        self.weights = np.zeros(X.shape[1]) # Random weight
        self.intercept = 0                  # Random intercept
        losses = []                         # Losses values for each lerning steps
        l1_penalty = 0.01                   # L1 regularization

        # For each epochs
        for _ in range(epochs):
            # Prediction calculation
            y_pred = self.predict(X)

            # Error calculation
            error = y_pred - y

            # MSE calculation
            loss = np.mean(np.square(error))
            losses.append(loss)

            # Gradient calculation
            gradient = np.dot(X.T, error) / len(y) + self.weights

            # L1 regularization to the gradient
            l1_gradient = l1_penalty * np.sign(self.weights)

            # Weights and intercept update with gradient descent
            self.weights = self.weights - learning_rate * (gradient + l1_gradient)
            self.intercept -= learning_rate * np.mean(error)

        return losses
    
    def fit_with_batch(self, X, y, learning_rate: float = 0.001, epochs: int = 1000):
        """! Gradient descent method with batch

        Function for regression using the gradient descente and random batch.

        @param X: Abscissa data
        @type X: Array

        @param X: Ordinate data
        @type X: Array

        @return: Predictions
        @rtype: Float

        """
        
        # Initialization of variables
        self.weights = np.zeros(X.shape[1]) # Random weight
        self.intercept = 0                  # Random intercept
        losses = []                         # Losses values for each lerning steps
        batch_size = 5                      # Number of batch

        for _ in range(epochs):
            # Shuffle the data indices
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)

            # Split data into mini-batches
            for i in range(0, X.shape[0], batch_size):
                # Select mini-batch
                batch_indices = indices[i:i+batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # Prediction calculation
                y_pred = self.predict(X_batch)

                # Error calculation
                error = y_pred - y_batch

                # MSE calculation
                loss = np.mean(np.square(error))
                losses.append(loss)

                # Gradient calculation
                gradient = np.dot(X_batch.T, error) / len(y_batch) + self.weights

                # Weights and intercept update with gradient descent
                self.weights -= learning_rate * gradient
                self.intercept -= learning_rate * np.mean(error)

        return losses

    def predict(self, X):
        """! Calulation of predicted value

        Function for calculate the prediction.

        @param X: Abscissa data
        @type X: Array

        @return: Predictions
        @rtype: Float

        """
            
        # Prediction calculating with weight 
        return X @ self.weights + self.intercept

    def plot_learning_curve(self, losses):
        """! Plot display

        Function to display the plot with learning curve.

        @param losses: Mean Squared Error
        @type losses: List

        """
            
        import matplotlib.pyplot as plt

        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.title('Learning Curve')
        plt.show()


def compute_mse(prediction, ground_truth):
    """! Calculation of total error (MSE)

    Function for calculate the total error (MSE).

    @param prediction: Predicted values
    @type prediction: Array

    @param ground_truth: True value
    @type ground_truth: Array

    @return: MSE
    @rtype: Float

    """
        
    # MSE calculation
    return np.mean(np.square(prediction - ground_truth))

def main():
    # Initialization of variables
    train_df = pd.read_csv('../data/train.csv')                             # Training data
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy() # Abscissa of training data
    train_y = train_df["Performance Index"].to_numpy()                # Ordinate of training data

    # Close form method
    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f'{LR_CF.weights = }, {LR_CF.intercept = :.4f}')

    # Gradient descent method
    LR_GD = LinearRegressionGradientdescent()
    losses = LR_GD.fit(train_x, train_y, learning_rate=1e-4, epochs=10000)
    LR_GD.plot_learning_curve(losses)
    logger.info(f'{LR_GD.weights = }, {LR_GD.intercept = :.4f}')

    # Initialization of variables
    test_df = pd.read_csv('../data/test.csv')                             # Test data
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy() # Abscissa of test data
    test_y = test_df["Performance Index"].to_numpy()                # Ordinate of test data

    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).sum()
    logger.info(f'Prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = abs(((mse_gd - mse_cf) / mse_cf) * 100)
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')

if __name__ == '__main__':
    main()
