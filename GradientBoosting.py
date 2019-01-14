#n_estimators - number of trained trees
#subsample - fraction of data to use on each stage
#learning_rate - size of step
#train_features - features used by tree
#depth - tree depth

import numpy as np
import pandas as pd
from datetime import datetime
import math
import operator

import xgboost as xgboost
from scipy.special import expit
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y
from sklearn.utils.random import check_random_state
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, mean_squared_error

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
X.head()
Y = Y.astype(float)
#Y

num_test = 0.20
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=num_test, random_state=23)


def _estimate_tree(leaf_values, X):
    """taking indices of leaves and return the corresponding value for each event"""
    leaves = transform(X, sigmoid)
    return leaf_values[leaves]

def transform(val, func):
    #Apply logistic func to the output
    return func(val)

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-1.0 * z))

def sigmoidGradient(y, y_pred):
    #return np.matmul(sigmoid(-actual*predicted), 1 - sigmoid(-actual*predicted))
    return y*sigmoid(-y*y_pred)
    #return sigmoid(-y*y_pred)*(1 - sigmoid(-y*y_pred))

def run_search(X, Y, clf, parameters):
        scorer = make_scorer(accuracy_score)
        grid_obj = GridSearchCV(clf, parameters, cv=KFold(n_splits=3),
                                scoring=scorer)
        grid_obj = grid_obj.fit(X, Y)
        return grid_obj.best_estimator_

def std_agg(cnt, s1, s2): return math.sqrt((s2/cnt) - (s1/cnt)**2)

class GradientBoosting:
    def __init__(self, n_estimators, subsample, train_features, depth, learning_rate=0.01, random_state=None):
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.learning_rate = learning_rate
        self.train_features = train_features
        self.depth = depth
        self.loss_func = None
        self.trees = []
        self.random_state = random_state

    def fit(self, X, y):
        #self.estimators = []
        #self.scores = []
        #self.n_features = X.shape[1]
        #x = X
        #yi = y
        X, y = check_X_y(X, y)
        n_samples = X.shape[0]
        n_inbag = int(n_samples/self.subsample)
        #for logistic loss convert labels from {0, 1} to {-1, 1}
        #y = (y * 2) - 1
        y_pred = np.zeros(n_samples)
        #y_pred = np.full(n_samples, np.mean(y))

        for n in range(self.n_estimators):
            #Least squares loss grad
            # theta = theta + alpha * grad - alpha * C * theta
            #error
            #residuals = std_agg(n_samples, y, y_pred)
            #residuals = sigmoidGradient(y, y_pred)

            #residuals = sigmoidGradient(y, y_pred)
            residuals = y - y_pred

            tree = DecisionTreeRegressor(criterion='mse',
                                         min_samples_leaf=5,
                                         max_features=self.train_features,
                                         max_depth=self.depth,
                                         min_samples_split=self.subsample)
            tree.fit(X, residuals)
            y_pred = tree.predict(X) + self.learning_rate*sigmoidGradient(y, y_pred)/n_samples - self.learning_rate*tree.predict(X)
            #Logistic loss
            #y * expit(-y* y_pred)

            #tree = DecisionTreeClassifier(criterion='gini',
            #                                               min_samples_leaf=5,
            #                                               max_features=self.train_features,
            #                                               max_depth=self.depth,
            #                                               min_samples_split=self.subsample)

            #best_clfs = {}

            #best_clfs['DecisionTreeClassifier'] = run_search(
             #   X, residuals, DecisionTreeClassifier(),
              #  {
              #      'criterion': ['entropy', 'gini'],
              #      'min_samples_leaf': [3, 5],
              #      'max_depth': [3, 4],
              #      'min_samples_split': [3, 5, 8]
              #  })
            #tree = best_clfs['DecisionTreeClassifier']

            #rand = check_random_state(self.random_state)
            #train_indices = rand.choice(n_samples, size=n_inbag, replace=False)
            #tree.fit(X, targets)

            #tree.fit(X, residuals)
            #aa = tree.apply(X[train_indices])
            #predictions = tree.predict(X)
            #y_pred += predictions
            #y_pred += self.learning_rate*sigmoidGradient(y, y_pred) - self.learning_rate*y_pred
            #var = _estimate_tree(tree.apply(X[train_indices]), X[train_indices])
            #theta = theta + alpha * grad - alpha * C * theta
            #y_pred = predictions + self.learning_rate * grad - self.learning_rate*predictions
            #y_pred = predictions - sigmoidGradient(y, predictions)
            self.trees.append(tree)

    #predict classes for each event
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])

        for i, tree in enumerate(self.trees):
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred


        #predict probabilities for each event
    def predict_probabilities(self, X):
        return transform(self.predict(X), sigmoid)

#class GradientBoostingClassifier(GradientBoosting):
 #   def fit(self, X, y=None):
 #       # Convert labels from {0, 1} to {-1, 1}
  #      y = (y * 2) - 1
  #      self.loss = LogisticLoss()
#super(GradientBoostingClassifier, self).fit(X, y)
#m, n = X.shape
#h = sigmoid(np.matmul(np.matmul(X, np.zeros(n)), Y.T))
#grad = np.matmul(Y, X * (1 - h)) / m
model = GradientBoosting(n_estimators=1000, depth=5, train_features='sqrt', subsample=5)

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
#predict_res = model.predict_probabilities(X_target)
predict_res = model.predict(X_target)
#print('regression, mse: %s'% mean_squared_error(y_test.flatten(), predictions.flatten()))
prediction['pickups'] = predict_res
prediction.to_csv('all/baseline.csv', index=None)


X_t = pd.read_csv('all/trainX.csv')
pred = pd.DataFrame()
pred['id'] = [str(item) for item in X_t['id']]
X_t = total_trasform(X_t)
pred_res = model.predict(X_t)
pred['pickups'] = pred_res
pred.to_csv('all/out.csv', index=None)