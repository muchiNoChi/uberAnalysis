import random

import numpy as np
from scipy import stats

class Tree(object):
    """Recursive implementation of decision tree."""

    def __init__(self, regression=False, criterion=None, max_features=None, min_samples_split=10, max_depth=None,
                 minimum_gain=0.01, loss=None):
        self.regression = regression
        self.impurity = None
        self.threshold = None
        self.column_index = None
        self.outcome = None
        self.criterion = criterion
        self.loss = loss
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.minimum_gain = minimum_gain

        self.left_child = None
        self.right_child = None

    @property
    def is_terminal(self):
        return not bool(self.left_child and self.right_child)

    def _find_splits(self, X):
        """Find all possible split values."""
        split_values = set()

        # Get unique values in a sorted order
        x_unique = list(np.unique(X))
        for i in range(1, len(x_unique)):
            # Find a point between two values
            average = (x_unique[i - 1] + x_unique[i]) / 2.0
            split_values.add(average)

        return list(split_values)

    def _find_best_split(self, X, target, n_features):
        """Find best feature and value for a split. Greedy algorithm."""

        # Sample random subset of features
        subset = random.sample(list(range(0, X.shape[1])), n_features)
        max_gain, max_col, max_val = None, None, None

        for column in subset:
            split_values = self._find_splits(X[:, column])
            for value in split_values:
                # Gradient boosting
                left, right = split_dataset(X, target, column, value, return_X=False)
                gain = xgb_criterion(target, left, right, self.loss)

                if (max_gain is None) or (gain > max_gain):
                    max_col, max_val, max_gain = column, value, gain
        return max_col, max_val, max_gain

    def fit(self, X, target):
        """Build a decision tree from training set.
        Parameters
        ----------
        X : array-like
            Feature dataset.
        target : dictionary or array-like
            Target values.
        """

        if not isinstance(target, dict):
            target = {'y': target}

        try:
            # Exit from recursion using assert syntax
            assert (X.shape[0] > self.min_samples_split)
            assert (self.max_depth > 0)

            if self.max_features is None:
                self.max_features = X.shape[1]

            column, value, gain = self._find_best_split(X, target, self.max_features)
            assert gain is not None
            if self.regression:
                assert (gain != 0)
            else:
                assert (gain > self.minimum_gain)

            self.column_index = column
            self.threshold = value
            self.impurity = gain

            # Split dataset
            left_X, right_X, left_target, right_target = split_dataset(X, target, column, value)

            # Grow left and right child
            self.left_child = Tree(self.regression, self.criterion)
            self.left_child.fit(left_X, left_target)

            self.right_child = Tree(self.regression, self.criterion)
            self.right_child.fit(right_X, right_target)
        except AssertionError:
            self.outcome = self.loss.approximate(target['actual'], target['y_pred'])

    def predict_row(self, row):
        """Predict single row."""
        if not self.is_terminal:
            if row[self.column_index] < self.threshold:
                return self.left_child.predict_row(row)
            else:
                return self.right_child.predict_row(row)
        return self.outcome

    def predict(self, X):
        result = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            result[i] = self.predict_row(X[i, :])
        return result


def split(X, y, value):
    left_mask = (X < value)
    right_mask = (X >= value)
    return y[left_mask], y[right_mask]


def split_dataset(X, target, column, value, return_X=True):
    left_mask, right_mask = X[:, column] < value, X[:, column] >= value

    left, right = {}, {}
    for key in target.keys():
        left[key] = target[key][left_mask]
        right[key] = target[key][right_mask]

    if return_X:
        left_X, right_X = X[left_mask], X[right_mask]
        return left_X, right_X, left, right
    else:
        return left, right


def xgb_criterion(y, left, right, loss):
    left = loss.gain(left['y'], left['y_pred'])
    right = loss.gain(right['y'], right['y_pred'])
    initial = loss.gain(y['y'], y['y_pred'])
    gain = left + right - initial
    return gain
