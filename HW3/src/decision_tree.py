import numpy as np
import pandas as pd


class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.feature_importances_ = None

    def fit(self, X, y):
        self.feature_importances_ = np.zeros(X.shape[1])
        self.tree = self._grow_tree(X, y)
        total_importance = self.feature_importances_.sum()
        if total_importance > 0:
            self.feature_importances_ /= total_importance

    def _grow_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(set(y)) == 1:
            return {'value': np.bincount(y).argmax()}
        feature_index, threshold, impurity_reduction = find_best_split(X, y, self.feature_importances_)
        if feature_index is None:
            return {'value': np.bincount(y).argmax()}
        left, right = split_dataset(X, y, feature_index, threshold)

        self.feature_importances_[feature_index] += impurity_reduction
        return {
            'feature_index': feature_index,
            'threshold': threshold,
            'left': self._grow_tree(left[0], left[1], depth + 1),
            'right': self._grow_tree(right[0], right[1], depth + 1),
        }

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return [self._predict_tree(x, self.tree) for x in X]

    def _predict_tree(self, x, tree_node):
        if 'value' in tree_node:
            return tree_node['value']
        feature_index = tree_node['feature_index']
        threshold = tree_node['threshold']
        if x[feature_index] <= threshold:
            return self._predict_tree(x, tree_node['left'])
        else:
            return self._predict_tree(x, tree_node['right'])


def split_dataset(X, y, feature_index, threshold):
    left_mask = X.iloc[:, feature_index] <= threshold
    right_mask = X.iloc[:, feature_index] > threshold
    return (X[left_mask], y[left_mask]), (X[right_mask], y[right_mask])


def find_best_split(X, y, feature_importances):
    best_feature, best_threshold, best_score = None, None, float('inf')
    best_impurity_reduction = 0
    current_impurity = entropy(y)
    for feature_index in range(X.shape[1]):
        thresholds = np.unique(X.values[:, feature_index])
        for threshold in thresholds:
            (X_left, y_left), (X_right, y_right) = split_dataset(X, y, feature_index, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            score = (
                len(y_left) / len(y) * entropy(y_left)
                + len(y_right) / len(y) * entropy(y_right)
            )
            impurity_reduction = current_impurity - score
            if score < best_score:
                best_score = score
                best_feature = feature_index
                best_threshold = threshold
                best_impurity_reduction = impurity_reduction
    return best_feature, best_threshold, best_impurity_reduction


def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    entropy_value = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy_value


def gini_index(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    gini = 1 - np.sum([p ** 2 for p in probabilities])
    return gini
