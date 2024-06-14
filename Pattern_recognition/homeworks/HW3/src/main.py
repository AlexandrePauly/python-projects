import numpy as np
import pandas as pd
from loguru import logger

from src import AdaBoostClassifier, BaggingClassifier, DecisionTree, gini, entropy
from src.utils import plot_learners_roc, accuracy_score, plot_feature_importance, best_prediction


def main():
    # Load data
    train_df = pd.read_csv('../data/train.csv')  # Training data
    test_df = pd.read_csv('../data/test.csv')    # Test data

    # Initialization of variables
    X_train = train_df.drop(['target'], axis=1).to_numpy()  # Training input
    y_train = train_df['target'].to_numpy()                 # Training target
    X_test = test_df.drop(['target'], axis=1).to_numpy()    # Test input
    y_test = test_df['target'].to_numpy()                   # Test target

    # Load feature names
    feature_names = list(train_df.drop(['target'], axis=1).columns)

    # AdaBoost
    clf_adaboost = AdaBoostClassifier(input_dim=X_train.shape[1])
    _ = clf_adaboost.fit(X_train, y_train, num_epochs=500, learning_rate=0.001)
    y_pred_classes, y_pred_probs = clf_adaboost.predict_learners(X_test)
    y_test[y_test == 0] = -1
    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'AdaBoost - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(y_preds=y_pred_probs, y_trues=y_test, fpath='./tmp_adaboost.png')

    # Ploting feature importances
    feature_importance = clf_adaboost.compute_feature_importance(X_test)
    plot_feature_importance(feature_importance=feature_importance, feature_names=feature_names)

    # Bagging
    clf_bagging = BaggingClassifier(input_dim=X_train.shape[1])
    _ = clf_bagging.fit(X_train, y_train, num_epochs=2000, learning_rate=0.0001)
    y_pred_classes, y_pred_probs = clf_bagging.predict_learners(X_test)
    y_test[y_test == -1] = 0
    y_pred_classes = best_prediction(y_test, y_pred_classes)
    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'Bagging - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(y_preds=y_pred_probs, y_trues=y_test, fpath='./tmp_bagging.png')

    # Ploting feature importances
    # feature_importance = clf_bagging.compute_feature_importance()
    # plot_feature_importance(feature_importance=feature_importance, feature_names=feature_names)

    # Gini and entropy
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    logger.info(f'Gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1] is : {gini(data):.4f}')
    logger.info(f'Entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1] is : {entropy(data):.4f}')

    # Reshape targets
    y_train = train_df['target'].to_numpy().reshape(-1, 1)
    y_test = test_df['target'].to_numpy().reshape(-1, 1)

    # Decision Tree
    clf_tree = DecisionTree(criterion=entropy, max_depth=7)
    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)
    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'DecisionTree - Accuracy: {accuracy_:.4f}')

    # Ploting feature importances
    feature_importance = clf_tree.compute_feature_importance(X_test)
    plot_feature_importance(feature_importance=feature_importance, feature_names=feature_names)


if __name__ == '__main__':
    import torch
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    main()
