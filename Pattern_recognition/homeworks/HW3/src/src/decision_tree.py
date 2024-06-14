import numpy as np


class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, quality=None, value=None):
        """! Constructor

        Function for generate the constructor.

        @param feature_index: Feature index
        @type feature_index: Integer

        @param threshold: Threshold value
        @type threshold: Float

        @param left: Left subset of the node
        @type left: Array

        @param right: Right subset of the node
        @type right: Array

        @param quality: Gain value
        @type quality: Float

        @param value: Leaf value
        @type value: Integer

        """

        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.quality = quality
        self.value = value


class DecisionTree:
    def __init__(self, criterion, max_depth=1):
        """! Constructor

        Function for generate the constructor.

        @param criterion: Criterion used (gini or entropy)
        @type criterion: String

        @param max_depth: Max depth
        @type max_depth: Integer

        """

        # Initialization of variables
        self.criterion = criterion  # Criterion
        self.max_depth = max_depth  # Max depth

    def fit(self, X, y):
        """! Model

        Function for generate the model.

        @param X: Features
        @type X: Array

        @param y: Target
        @type y: Array

        """

        # Dataset concatenation
        dataset = np.concatenate((X, y), axis=1)

        # Decision tree calculation
        self.tree = self._grow_tree(dataset)

    def compute_feature_importance(self, X):
        """! Feature importance

        Function for calculate the feature importance.

        @param X: Features
        @type X: Array

        @return: Feature importances
        @rtype: Array

        """

        # Initialize an array to store feature importances
        feature_importance = np.zeros(X.shape[1])

        # Recursive function to traverse the tree and accumulate feature importance
        def traverse_tree(node, importance):
            if node.left is None and node.right is None:
                # Leaf node, return importance
                return importance
            else:
                # Non-leaf node, accumulate importance
                importance[node.feature_index] += node.quality
                importance = traverse_tree(node.left, importance)
                importance = traverse_tree(node.right, importance)
                return importance

        # Call the recursive function to accumulate feature importance
        feature_importance = traverse_tree(self.tree, feature_importance)

        # Normalize feature importances to sum up to 1
        feature_importance /= np.sum(feature_importance)

        return feature_importance

    def _grow_tree(self, dataset, depth=1):
        """! Decision tree calculation

        Function for calculate the decision tree.

        @param dataset: Dataset (X and y)
        @type dataset: Array

        @param depth: Depth
        @type depth: Integer

        @return: Node(value=leaf_value)
        @rtype: value

        """

        # Initialization of variables
        X, y = dataset[:, :-1], dataset[:, -1]  # Split the dataset in input and target values
        num_samples = np.shape(X)[0]            # Number of samples

        # Set the number of samples as the max_depth if max_depth is not define
        if self.max_depth is None:
            self.max_depth = num_samples

        # If the current depth is within the maximum depth limit
        if depth <= self.max_depth:
            # Find the best split for the current dataset
            best_split = find_best_split(dataset, self.criterion)

            # If a valid split is found
            if len(best_split) != 0:
                # If the quality of the split is greater than 0 (indicating impurity reduction)
                if best_split["quality"] > 0:
                    # Recursively grow the left subtree
                    left_subtree = self._grow_tree(best_split["left"], depth + 1)

                    # Recursively grow the right subtree
                    right_subtree = self._grow_tree(best_split["right"], depth + 1)

                    # Create a node representing the current split
                    return Node(best_split["feature_index"], best_split["threshold"],
                                left_subtree, right_subtree, best_split["quality"])

        # Leaf node
        leaf_value = np.argmax(np.bincount(y))

        return Node(value=leaf_value)

    def predict(self, X):
        """! Calulation of predicted value

        Function for calculate the prediction.

        @param X: Data
        @type X: Array

        @return: Predictions
        @rtype: Array

        """

        preditions = [self._predict_tree(v, self.tree) for v in X]

        return preditions

    def _predict_tree(self, x, tree_node):
        """! Calulation of predicted value

        Function for predict the target value for a given input data point using the decision tree.

        @param x: Data
        @type x: Array

        @param tree_node: Data
        @type tree_node: Node object

        @return: Predictions
        @rtype: Array

        """

        # If the current node is a leaf node (i.e., it has a value assigned), return that value
        if tree_node.value is not None:
            return tree_node.value

        # Extract the feature value for the current node from the input data point
        feature_val = x[tree_node.feature_index]

        # If the feature value is less than or equal to the threshold, traverse the left subtree
        if feature_val <= tree_node.threshold:
            return self._predict_tree(x, tree_node.left)
        # Otherwise, traverse the right subtree
        else:
            return self._predict_tree(x, tree_node.right)


def split_dataset(X, y, feature_index, threshold):
    """! Split dataset

    Function for split the dataset in left and right subset based on a specified feature and threshold.

    @param X: Data
    @type X: Array

    @param y: Target
    @type y: Array

    @param feature_index: Index of the feature to split
    @type feature_index: Integer

    @param threshold: Value for splitting the feature
    @type threshold: String

    @return: Left and right subset
    @rtype: Array and Array

    """

    # Preprocessing data
    y = y.reshape(-1, 1)
    dataset = np.concatenate((X, y), axis=1)

    # Split the dataset in left and right subset based on the specified feature and threshold
    left = np.array([row for row in dataset if row[feature_index] <= threshold])
    right = np.array([row for row in dataset if row[feature_index] > threshold])

    return left, right


def find_best_split(dataset, criterion):
    """! Find best split

    Function to find the best split for the dataset.

    @param dataset: Data (Input and target)
    @type dataset: Array

    @param criterion: Criterion used (entropy or gini)
    @type criterion: Function

    @return: Best split
    @rtype: Dict

    """

    # Initialization of variables
    X, y = dataset[:, :-1], dataset[:, -1]             # Separate features and target
    best_split = {}                                    # To store information about the best split
    max_quality = -float("inf")                        # To be sure to update the best split
    feature_idx_list = [i for i in range(X.shape[1])]  # List of all features index

    # For each features index
    for feature_index in feature_idx_list:
        # Extract the values of the current feature
        feature_values = dataset[:, feature_index]

        # Get unique values of the feature to consider as potential thresholds for splitting
        possible_thresholds = np.unique(feature_values)

        # List to store thresholds values
        thresholds = []

        # Generate potential thresholds by taking the midpoint between consecutive unique feature values
        for t in range(len(possible_thresholds) - 1):
            new = (possible_thresholds[t] + possible_thresholds[t + 1]) / 2
            thresholds.append(new)

        # For each threshold
        for threshold in thresholds:
            # Split the dataset based on the current feature and threshold
            left, right = split_dataset(X, y, feature_index, threshold)

            # Ensure that both resulting subsets have at least one data point
            if len(left) > 0 and len(right) > 0:
                # Separate target values for the left and right subsets
                left_y, right_y = left[:, -1], right[:, -1]

                # Calculate the quality of this split
                current_gain = gain(y, left_y, right_y, criterion)

                # Update the best split if the current split has higher quality
                if current_gain > max_quality:
                    best_split["feature_index"] = feature_index
                    best_split["threshold"] = threshold
                    best_split["left"] = left
                    best_split["right"] = right
                    best_split["quality"] = current_gain
                    max_quality = current_gain

    return best_split


def gain(parent, left_child, right_child, mode):
    """! Calulation of gain

    Function for calculate the gain/quality.

    @param parent: Data of node parent
    @type parent: Array

    @param left_child: Data of left child
    @type left_child: Array

    @param right_child: Data of right child
    @type right_child: Array

    @param mode: Entropy or Gini
    @type mode: String

    @return: Gain
    @rtype: Float

    """

    weight_left = len(left_child) / len(parent)
    weight_right = len(right_child) / len(parent)

    # Gain calculation
    gain = mode(parent) - (weight_left * mode(left_child) + weight_right * mode(right_child))

    return gain


def entropy(y):
    """! Calulation of entropy

    Function for calculate the entropy.

    @param y: Data
    @type y: Array

    @return: Entropy
    @rtype: Array

    """

    n = len(y)
    _, count = np.unique(y, return_counts=True)
    prob = count / n

    entropy = -1 * np.sum(prob * np.log2(prob))

    return entropy


def gini(y):
    """! Calulation of gini index

    Function for calculate the gini index.

    @param y: Data
    @type y: Array

    @return: Gini index
    @rtype: Array

    """

    n = len(y)
    _, count = np.unique(y, return_counts=True)
    prob = count / n

    gini = 1 - np.sum(prob ** 2)

    return gini
