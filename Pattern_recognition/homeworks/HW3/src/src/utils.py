import typing as t
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_curve, auc


class WeakClassifierAdaboost(nn.Module):
    def __init__(self, input_dim):
        """! Constructor

        Function for generate the constructor.

        @param input_dim: Input dimension
        @type input_dim: Array

        """

        pass

    def forward(self, X, threshold, direction, feature_index):
        feature_values = X[:, feature_index]
        predictions = np.ones(X.shape[0])
        predictions[feature_values * direction < threshold * direction] = -1

        return predictions


class WeakClassifierBagging(nn.Module):
    def __init__(self, input_dim):
        """! Constructor

        Function for generate the constructor.

        @param input_dim: Input dimension
        @type input_dim: Array

        """

        super(WeakClassifierBagging, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def entropy_loss(outputs, targets):
    """! Entropy loss

    Function to calculte the entropy loss.

    @param outputs: Outputs of traning model
    @type outputs: Array

    @param targets: True labels
    @type targets: Array

    @return: Loss by entropy
    @rtype: Float

    """

    # For numerical stability
    epsilon = 1e-10
    prob = torch.sigmoid(outputs)
    loss = -torch.mean(targets * torch.log(prob + epsilon) + (1 - targets) * torch.log(1 - prob + epsilon))

    return loss


def plot_learners_roc(y_preds: t.List[t.Sequence[float]], y_trues: t.Sequence[int], fpath='./tmp.png'):
    """! Plotting learners ROC

    Function for generate and plotting learners ROC.

    @param y_preds: Predictions
    @type y_preds: Array

    @param y_trues: Target
    @type y_trues: Array

    @param fpath: File path to store plot
    @type fpath: String

    """

    plt.figure(figsize=(8, 6))
    for i, y_pred in enumerate(y_preds):
        fpr, tpr, _ = roc_curve(y_trues, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Learner {i+1} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(fpath)
    plt.show()


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


def plot_feature_importance(feature_importance, feature_names):
    """! Plot feature importance

    Function to plot feature importances

    @param feature_importances: Feature importances
    @type feature_importances: Array

    @param feature_names: Feature names
    @type feature_names: Array

    """

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance)), feature_importance, align='center')
    plt.yticks(range(len(feature_importance)), feature_names)
    plt.title('Feature importance')
    plt.show()


def best_prediction(y_test, y_pred_classes):
    """! Best prediction

    Function to keep the best prediction for a classifier.

    @param y_test: Targets
    @type y_test: Array

    @param y_pred_classes: Predictions
    @type y_pred_classes: Array

    @return: Best prediction
    @rtype: Array

    """

    # Initialization of variables
    best_class_pred = []  # List to store best predictions
    best_accuracy = 0     # Float to store best accuracy

    # For each prediction
    for i in range(len(y_pred_classes)):
        if accuracy_score(y_test, y_pred_classes[i]) > best_accuracy:
            best_accuracy = accuracy_score(y_test, y_pred_classes[i])
            best_class_pred = y_pred_classes[i]

    return best_class_pred
