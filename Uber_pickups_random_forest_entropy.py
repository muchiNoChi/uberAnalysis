#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import math
import operator

from datetime import datetime


# # Loading the data

# For this problem '*Uber Pickups*' dataset was chosen. It represents the data for different New York Uber pickups.
# Our goal is, by analysing this dataset, to predict the amount of pickups for each borough presented.

# In[ ]:


X_data = pd.read_csv('data/trainX.csv')
Y_data = pd.read_csv('data/trainY.csv')


# In[ ]:


X_data.head()


# For a solution using scikit-learn we considered 3 types of classifiers: *RandomForest*, *DecisionTree* and *LogisticRegression*.
# 
# (See solution using scikit-learn for details)
# 
# The following results (in points) were achieved by fitting (after applying Grid Search with different parameters on those classifiers (specific solvers were chosen)):
# 
# *   RandomForest (entropy) - **1246401.3858999142**
# *   DecisionTree (gini) - **1245676.0787156357**
# *   LogisticRegression (newton-cg) - **1193406.0331829898**
# 
# According to these results, the best algorithm to apply on our dataset (between those 3 been tried) is ***RandomForest***.
# It will be implemented in this work to be then applied.

# # Preparing the data

# First, we don't need a pickup **id**.

# In[ ]:


X = pd.DataFrame(X_data, copy=True)
del X['id']


# Second, we'll transform **pickup_dt** feature into *numeric* type (as done in a solution using scikit-learn).

# In[ ]:


X['pickup_dt'] = X['pickup_dt'].apply(
    lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S").time()
).apply(
    lambda x: x.second+x.minute*60+x.hour*3600
)


# Third, let's transform **holiday** char typed feature into *binary* typed

# In[ ]:


X['hday'] = X['hday'].apply(lambda x: 0 if x == 'N' else 1)


# In[ ]:


X.head()


# Now let's normalize the data. We will use numpy's *linalg.norm()* function. It won't be implemented here as it's big enough, and you always can take a look at the source code via [this link](https://github.com/numpy/numpy/blob/v1.15.1/numpy/linalg/linalg.py#L2203-L2440).

# In[ ]:


def normalize(v):
  norm = np.linalg.norm(v)
  if norm == 0:
    return v
  return v / norm


# In[ ]:


for column_name in ['pickup_dt', 'spd', 'vsb', 'temp', 'dewp', 
                    'slp', 'pcp01', 'pcp06', 'pcp24', 'sd']:
  X[column_name] = normalize(X[column_name])


# In[ ]:


X.head()


# Data-binning for categorizing continous values:

# In[ ]:


for column_name in ['pickup_dt', 'spd', 'vsb', 'temp', 'dewp', 
                    'slp', 'pcp01', 'pcp06', 'pcp24', 'sd']:
    bins = []
#     we will create 20 bins to try
    step = (X[column_name].max() - X[column_name].min()) / 20
    for i in range(21):
        bins.append(X[column_name].min() + (step * i))
    binned_column_name = '{}_binned'.format(column_name)
    X[binned_column_name] = pd.cut(X[column_name], bins=bins)
X.head()


# Let's aggregate amount of values for bins and group them by size:

# In[ ]:


for column_name in ['pickup_dt', 'spd', 'vsb', 'temp', 'dewp', 
                    'slp', 'pcp01', 'pcp06', 'pcp24', 'sd']:
    bins = []
    step = (X[column_name].max() - X[column_name].min()) / 20
    for i in range(21):
        bins.append(X[column_name].min() + (step * i))
    binned = pd.cut(X[column_name], bins=bins).value_counts()
    print(binned)


# As it was done in a solution using scikit-learn, let's apply *one-hot encoding* for transforming **borough** feature data:

# In[ ]:


set(X['borough'].values)


# In[ ]:


for value in set(X['borough'].values):
    new_value = 'borough_{}'.format(value)
    X[new_value] = (X['borough'] == value)


# In[ ]:


del X['borough']
del X['borough_nan']
X.head()


# Let's prepare the vector of answers: 

# In[ ]:


Y_data.head()


# In[ ]:


merged = pd.merge(X_data, Y_data, on=['id'])
Y = merged['pickups'].values


# Now let's split our training dataset into **N** pieces. We will then train **N** *decision trees* on each subset respectively. This will allow us to create an *ensemble* to improve results and minimize the possibility of overfitting.
# 
# We'll try to take **N** = 10.

# In[ ]:


X_copy = pd.DataFrame(X, copy=True)
subsets = []
amount = int(len(X_copy) / 10)

for x in range(0, 10):
    subsets.append(X_copy.sample(n=amount))
    X_copy.drop(subsets[x].index, inplace=True)


# Now we have **10** random **subsets** formed from the original set. The next step is to implement the classifier. 

# # Implementing Decision Tree (entropy)

# *GridSearch* that was used in a solution using scikit-learn showed that *entropy-based* solver gives better results for our problem than *gini-based*. So ***entropy-based*** solver will be used here.

# In[ ]:


def entropy(data, attr):
#     storing the amount of each value (use bins for continous values here - TODO)
    ser = data.groupby(attr).size().to_dict()
    
    entries = len(data)
    entropy = 0.0  # default value
    for key in ser.keys():
#         counting the probability of the value to occur
        probability = float(ser[key])/entries
#     counting the entropy of the value by known formula
        entropy -= probability * math.log(probability,2)
    return entropy


# In[ ]:


entropy(X, 'borough_Bronx')


# In[ ]:


def split(data, colname, value):
#     returns dataframe without given colname which contains only given value of this colname
    return data.loc[data[colname] == value, data.columns.drop(colname)]


# In[ ]:


def choose(data):
#     helps to choose the best attribute for classification

#     TODO improve to use bins!
# using infogain term here - may be wrong, for continous data it's recommended to use
# gain_ratio, gain_ratio(data, attr) = gain(data, attr) / split_info(data, attr), where
# gain (data, attr) = entropy(data)(that means taking target attr as attr argument) - entropy(data, attr) and
# split_info(data, attr) = entropy(data, attr). TODO

    minimum_entropy = entropy(data, data.columns[0])
    best_attr = -1
    
    values_map = {col: data.groupby(col).size().to_dict() for col in data.columns}
    
    entropies = []
    
    for attr in values_map:
        new_entropy = 0.0
        for value in values_map[attr]:
#             data without attr with specific value of this attr only 
            new_data = split(X_copy, attr, value)
            probability = new_data.shape[0]/float(data.shape[0])
            new_entropy += probability * entropy(data, attr)
        entropies.append(dict(attr=attr, info_amount=new_entropy))
    
    print(entropies)
#     takes attr with minimum entropy as the best one
    best_attr = min(entropies, key=operator.itemgetter('info_amount'))
    print(f'best attribute is now {best_attr}')
    return best_attr['attr']


# In[ ]:


X_copy = pd.DataFrame(X, copy=True)


# In[ ]:


choose(X_copy)


# In[ ]:


def majority(classset):
#     returns the class that has the most votes
# TODO change implementation for regressor! use bins!
    count = {}
    for attr in classset:
        if vote not in count.keys(): count[vote] = 0
        count[vote] += 1
    sorted_class_count = sorted(count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


# In[ ]:


def tree(data, labels):
#     if only one entry - return it
    if data.shape[0] == 1:
        return data.loc[:0]
#     if only one column left - return the majority of it's values (TODO improve function)
    if data.columns.size == 1:
        return majority(data)
#     else choose best feat, create a node, go recursively.
    best_feat = choose(data)
    print(labels)
    the_tree = {best_feat:{}}
    labels.remove(best_feat)
    print(best_feat)
    feat_values = data[best_feat]
    unique_vals = feat_values.unique()
    for value in unique_vals:
        print(f'\n {value} ({len(labels)} feats remaining)')
        sublabels = labels.copy()
        the_tree[best_feat][value] = tree(split(data, best_feat, value), sublabels)
    return the_tree


# In[ ]:


X_copy = pd.DataFrame(X, copy=True)


# In[ ]:


tree(X_copy, set(X_copy))


# In[ ]:


entropy_map = {}
for feat in set(X_copy):
  entropy_map[feat] = entropy(X_copy, feat)


# In[ ]:


entropy_map


# In[ ]:




