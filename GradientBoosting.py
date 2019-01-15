#n_estimators - number of trained trees
#subsample - fraction of data to use on each stage
#learning_rate - size of step
#train_features - features used by tree
#depth - tree depth
import random

import numpy as np
import pandas as pd
from datetime import datetime
import math

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y
from sklearn.metrics import make_scorer, accuracy_score

import BaseTree

random.seed(111)

X_data = pd.read_csv('all/trainX.csv')
Y_data = pd.read_csv('all/trainY.csv')

X_data.head()


X = pd.DataFrame(X_data, copy=True)
del X['id']

X['pickup_dt'] = X['pickup_dt'].apply(
    lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S").time()
).apply(
    lambda x: x.second+x.minute*60+x.hour*3600
)


X['hday'] = X['hday'].apply(lambda x: 0 if x == 'N' else 1)


X.head()


def normalize(v):
  norm = np.linalg.norm(v)
  if norm == 0:
    return v
  return v / norm

set(X['borough'].values)

for value in set(X['borough'].values):
    new_value = 'borough_{}'.format(value)
    X[new_value] = (X['borough'] == value)

del X['borough']
del X['borough_nan']
X.head()

Y_data.head()

merged = pd.merge(X_data, Y_data, on=['id'])
Y = merged['pickups'].values

X = X.astype(float)
Y = Y.astype(float)

num_test = 0.20
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=num_test, random_state=23)


def _estimate_tree(leaf_values, X):
    """taking indices of leaves and return the corresponding value for each event"""
    leaves = transform(X, LogisticLoss.sigmoid)
    return leaf_values[leaves]


def transform(val, func):
    """Apply logistic func to the output"""
    return func(val)


class LogisticLoss:
    def __init__(self, regularization=1.0):
        self.regularization = regularization

    @staticmethod
    def sigmoid(z):
        return 1.0/(1.0 + np.exp(-1.0 * z))

    @staticmethod
    def gradient(y, y_pred):
        #return np.matmul(sigmoid(-actual*predicted), 1 - sigmoid(-actual*predicted))
        #return y*LogisticLoss.sigmoid(-y*y_pred)
        return LogisticLoss.sigmoid(-y*y_pred)*(1 - LogisticLoss.sigmoid(-y*y_pred))

    def approximate(self, y, y_pred):
        """Approximate leaf value."""
        return self.gradient(y, y_pred).sum() / (self.hessian(y_pred).sum() + self.regularization)

    def hessian(self, y_pred):
        return self.sigmoid(y_pred) * (1 - self.sigmoid(y_pred))

    def gain(self, y, y_pred):
        """Calculate gain for split search."""
        nominator = self.gradient(y, y_pred).sum() ** 2
        denominator = (self.hessian(y_pred).sum() + self.regularization)
        return 0.5 * (nominator / denominator)


class ListSqLoss:
    def __init__(self, regularization=1.0):
        self.regularization = regularization

    @staticmethod
    def gradient(y, y_pred):
        return y - y_pred

    def hessian(self, y):
        return np.ones_like(y)

    def approximate(self, y, y_pred):
        """Approximate leaf value."""
        return self.gradient(y, y_pred).sum() / (self.hessian(y_pred).sum() + self.regularization)

    def gain(self, y, y_pred):
        """Calculate gain for split search."""
        nominator = self.gradient(y, y_pred).sum() ** 2
        denominator = (self.hessian(y).sum() + self.regularization)
        return 0.5 * (nominator / denominator)


def run_search(X, Y, clf, parameters):
        scorer = make_scorer(accuracy_score)
        grid_obj = GridSearchCV(clf, parameters, cv=KFold(n_splits=3),
                                scoring=scorer)
        grid_obj = grid_obj.fit(X, Y)
        return grid_obj.best_estimator_


class GradientBoosting:
    def __init__(self, n_estimators, loss_func, train_features=None, subsample=1, depth=6, tree=None, eta=0.2,
                 random_state=None):
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.eta = eta
        self.train_features = train_features
        self.depth = depth
        self.loss_func = loss_func
        self.trees = []
        self.random_state = random_state
        self.tree = tree

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        #y_pred = np.full(n_samples, np.mean(y))

        for n in range(self.n_estimators):
            residuals = self.loss_func.gradient(y, y_pred)
            if self.tree is None:
                new_tree = DecisionTreeRegressor(criterion='mse', min_samples_leaf=5, max_features=self.train_features,
                                                 max_depth=self.depth, min_samples_split=self.subsample)
            else:
                new_tree = self.tree
            new_tree.fit(X, residuals)
            y_pred = y_pred + self.eta*new_tree.predict(X)
            self.trees.append(new_tree)

    """predict classes for each event"""
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])

        for i, tree in enumerate(self.trees):
            y_pred += self.eta * tree.predict(X)
        return y_pred

    """predict probabilities for each event"""
    def predict_probabilities(self, X, loss_func):
        return transform(self.predict(X), loss_func)


model = GradientBoosting(n_estimators=100, depth=5, train_features='sqrt', subsample=5, loss_func=ListSqLoss)

model.fit(X, Y)


def total_trasform(X):
    X['pickup_dt'] = X['pickup_dt'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S").time())
    X['pickup_dt'] = X['pickup_dt'].apply(lambda x: x.second+x.minute*60+x.hour*3600)
    X['hday'] = X['hday'].apply(lambda x: 0 if x == 'N' else 1)
    for column_name in ['pickup_dt', 'spd', 'vsb', 'temp', 'dewp',
                        'slp', 'pcp01', 'pcp06', 'pcp24', 'sd']:
        X[column_name] = normalize(X[column_name])
    for value in set(X['borough'].values):
        new_value = 'borough_{}'.format(value)
        X[new_value] = (X['borough'] == value)
    del X['borough']
    del X['borough_nan']
    del X['id']
    return X


X_target = pd.read_csv('all/testX.csv')
prediction = pd.DataFrame()
prediction['id'] = [str(item) for item in X_target['id']]
X_target = total_trasform(X_target)
predict_res = model.predict(X_target)
prediction['pickups'] = predict_res
prediction.to_csv('all/baseline.csv', index=None)


def mse_criterion(y, splits):
    y_mean = np.mean(y)
    return -sum([np.sum((split - y_mean) ** 2) * (float(split.shape[0]) / y.shape[0]) for split in splits])


base_tree = BaseTree.Tree(regression=True, criterion=mse_criterion, loss=ListSqLoss, max_depth=5, max_features=4)
model_2 = GradientBoosting(n_estimators=200, loss_func=ListSqLoss, tree=base_tree)
model_2.fit(X, Y)

X_target_2 = pd.read_csv('all/testX.csv')
prediction_2 = pd.DataFrame()
prediction_2['id'] = [str(item) for item in X_target_2['id']]
X_target_2 = total_trasform(X_target_2)
predict_res_2 = model_2.predict(X_target_2)
prediction_2['pickups'] = predict_res_2
prediction.to_csv('all/baseline_2.csv', index=None)
