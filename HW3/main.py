import pandas as pd
from loguru import logger
import random
import numpy as np
import torch
from src import AdaBoostClassifier, BaggingClassifier, DecisionTree
from src.utils import preprocess, plot_learners_roc, plot_feature_importance
from src.decision_tree import entropy, gini_index


def main():
    random.seed(777)  # DON'T CHANGE THIS LINE
    torch.manual_seed(777)  # DON'T CHANGE THIS LINE
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    X_train = train_df.drop(['target'], axis=1)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    X_test = test_df.drop(['target'], axis=1)
    y_test = test_df['target'].to_numpy()

    # (TODO): Implement you preprocessing function.
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    # AdaBoost
    clf_adaboost = AdaBoostClassifier(
        input_dim=X_train.shape[1],
    )
    _ = clf_adaboost.fit(
        X_train,
        y_train,
        num_epochs=10,
        learning_rate=1e4,
    )
    y_pred_classes, y_pred_probs = clf_adaboost.predict_learners(X_test)
    accuracy_ = np.sum(y_pred_classes == y_test) / len(y_test)
    logger.info(f'AdaBoost - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath="./Adaboost_roc.png",
    )
    feature_importance = clf_adaboost.compute_feature_importance()
    # Plot Bagging feature importance
    plot_feature_importance(
        feature_names=X_train.columns.tolist(),
        importance_values=feature_importance,
        title="Adaboost Feature Importance",
        fpath="./Adaboost_feature_importance.png"
    )

    # Bagging
    clf_bagging = BaggingClassifier(
        input_dim=X_train.shape[1],
    )
    _ = clf_bagging.fit(
        X_train,
        y_train,
        num_epochs=10,
        learning_rate=1e4,
    )
    y_pred_classes, y_pred_probs = clf_bagging.predict_learners(X_test)
    accuracy_ = np.sum(y_pred_classes == y_test) / len(y_test)
    logger.info(f'Bagging - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath="./Bagging_roc.png",
    )
    feature_importance = clf_bagging.compute_feature_importance()
    plot_feature_importance(
        feature_names=X_train.columns.tolist(),
        importance_values=feature_importance,
        title="Bagging Feature Importance",
        fpath="./Bagging_feature_importance.png"
    )

    # Decision Tree
    y_example = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"Entropy: {entropy(y_example):.4f}")
    print(f"Gini Index: {gini_index(y_example):.4f}")

    clf_tree = DecisionTree(
        max_depth=7,
    )
    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)
    accuracy_ = np.sum(y_pred_classes == y_test) / len(y_test)
    logger.info(f'DecisionTree - Accuracy: {accuracy_:.4f}')
    plot_feature_importance(
        feature_names=X_train.columns.tolist(),
        importance_values=clf_tree.feature_importances_,
        title="Decision Tree Feature Importance",
        fpath="./DecisionTree_feature_importance.png"
    )


if __name__ == '__main__':
    main()
