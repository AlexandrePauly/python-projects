import typing as t
import numpy as np
import torch
import torch.optim as optim
from .utils import WeakClassifierBagging, entropy_loss


class BaggingClassifier:
    def __init__(self, input_dim: int) -> None:
        """! Constructor

        Function for generate the constructor.

        @param input_dim: Input dimension
        @type input_dim: Array

        """

        # Create 10 learners
        self.learners = [
            WeakClassifierBagging(input_dim=input_dim) for _ in range(10)
        ]

    def fit(self, X_train, y_train, num_epochs: int, learning_rate: float):
        """! Model

        Function for generate the model.

        @param X_train: Features
        @type X_train: Array

        @param y_train: Target
        @type y_train: Array

        @param num_epochs: Number of epochs
        @type num_epochs: Integer

        @param learning_rate: Learning rate
        @type learning_rate: Float

        """

        # Initialization of variable
        losses_of_models = []       # List to store loss
        y_train[y_train == -1] = 0

        # For each weak learner
        for model in self.learners:
            # Convert data to PyTorch tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

            # Define optimizer
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

            # Model training
            model.train()

            # For each epoch
            for _ in range(num_epochs):
                # Forward
                outputs = model(X_train_tensor)
                loss = entropy_loss(outputs, y_train_tensor)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Store the loss
            losses_of_models.append(loss.item())

        return losses_of_models

    def predict_learners(self, X) -> t.Sequence[t.Sequence[t.Union[int, float]]]:
        """! Calulation of prediction using each weak learner

        Function for predict the target value for a given input data for each weak learner.

        @param x: Data
        @type x: Array

        @return: Predicted classes and probabilities
        @rtype: Array and array

        """

        # Initialization of variables
        predicted_classes = []  # List to store predicted classes
        predicted_probs = []    # List to store predicted probabilities

        # For each weak learner
        for model in self.learners:
            # Convert data to PyTorch tensor
            X_tensor = torch.tensor(X, dtype=torch.float32)

            # Make predictions
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor)
                probs = torch.sigmoid(outputs).numpy()

                # Apply threshold for a binary classification
                classes = (probs > 0.5).astype(int)
                predicted_classes.append(classes)
                predicted_probs.append(probs)

        return predicted_classes, predicted_probs

    def compute_feature_importance(self) -> t.Sequence[float]:
        feature_counts = np.zeros(self.learners[0].input_dim)  # Initialize feature counts

        # Count how many times each feature is used in the weak classifiers
        for model in self.learners:
            feature_counts += model.feature_usage

        # Normalize feature counts
        feature_importance = feature_counts / len(self.learners)

        return feature_importance.tolist()
