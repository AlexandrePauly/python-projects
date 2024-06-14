import typing as t
import numpy as np
from .utils import WeakClassifierAdaboost


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        """! Constructor

        Function for generate the constructor.

        @param input_dim: Input dimension
        @type input_dim: Array

        @param num_learners: Number of learners
        @type num_learners: Array

        """

        self.sample_weights = None
        self.learners = []
        self.num_learners = num_learners
        self.alphas = []

    def fit(self, X_train, y_train, num_epochs: int = 500, learning_rate: float = 0.001):
        y_train[y_train == 0] = -1
        n_samples = X_train.shape[0]
        self.sample_weights = np.ones(n_samples) / n_samples

        for _ in range(self.num_learners):
            weak_classifier = WeakClassifierAdaboost(input_dim=X_train.shape[1])
            min_error = float('inf')
            best_threshold = 0.0
            best_direction = 1.0
            best_feature_index = 0

            for feature_index in range(X_train.shape[1]):
                feature_values = X_train[:, feature_index]
                thresholds = np.unique(feature_values)
                for threshold in thresholds:
                    for direction in [1.0, -1.0]:
                        predictions = weak_classifier.forward(X_train, threshold, direction, feature_index)
                        error = np.sum(self.sample_weights[predictions != y_train])
                        if error < min_error:
                            min_error = error
                            best_threshold = threshold
                            best_direction = direction
                            best_feature_index = feature_index

            predictions = weak_classifier.forward(X_train, best_threshold, best_direction, best_feature_index)
            error = np.sum(self.sample_weights[predictions != y_train])
            alpha = 0.5 * np.log((1.0 - error) / max(error, 1e-10))
            self.alphas.append(alpha)
            self.learners.append((best_threshold, best_direction, best_feature_index))
            self.sample_weights *= np.exp(-alpha * y_train * predictions)
            self.sample_weights /= np.sum(self.sample_weights)

    def predict_learners(self, X) -> t.Tuple[t.Sequence[int], t.Sequence[float]]:
        """! Calulation of prediction using each weak learner

        Function for predict the target value for a given input data for each weak learner.

        @param x: Data
        @type x: Array

        @return: Predicted classes and probabilities
        @rtype: Array and array

        """

        # Initialization of variables
        predicted_probs = []                      # List to store predicted probabilities
        predicted_classes = np.zeros(X.shape[0])  # Predicted classes

        # For each wek learner
        for alpha, classifier in zip(self.alphas, self.learners):
            threshold, direction, feature_index = classifier
            predictions = alpha * WeakClassifierAdaboost(X.shape[1]).forward(X, threshold, direction, feature_index)
            predicted_classes += predictions
            proba = 1 / (1 + np.exp(-predicted_classes))
            predicted_probs.append(proba)

        predicted_classes = np.sign(predicted_classes)

        return predicted_classes, predicted_probs

    def compute_feature_importance(self, X) -> t.Sequence[float]:
        """! Feature importance

        Function for calculate the feature importance.

        @param X: Features
        @type X: Array

        @return: Feature importances
        @rtype: Array

        """

        # Initialize feature importance array
        feature_importance = np.zeros(X.shape[1])

        # For each weak classifier
        for alpha, model in zip(self.alphas, self.learners):
            _, _, feature_index = model

            # Accumulate importance based on alpha
            feature_importance[feature_index] += alpha

        # Normalize feature importance
        feature_importance /= np.sum(self.alphas)

        return feature_importance.tolist()
