#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2

from __future__ import division
import math
import numpy as np
import pandas as pd
import scipy
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from nolearn.dbn import DBN
from sklearn.cross_validation import *
from sklearn.decomposition import *
from sklearn.ensemble import *
from sklearn.externals import joblib
from sklearn.feature_selection import *
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.pipeline import *
from sklearn.preprocessing import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn import metrics
np.random.seed(42)

data = pd.read_csv('games.csv')._get_numeric_data()
Y = data[['TeamGoal+/-']].values
Y[Y<0] = 0
Y[Y>0] = 1
Y = np.array(Y)
Y = Y.reshape(Y.shape[0])

data.drop(['Unnamed: 0', 'TeamGoalFor', 'TeamGoalAgn', 'TeamGoal+/-', 'TeamGoal%', 'TeamGoalFor60', 'TeamGoalAgn60'], axis=1, inplace=True)
X = np.array(data)
print 'X.shape: ', X.shape
print 'Y.shape: ', Y.shape
print 'Columns: ', list(data.columns.values)

# filter rows with NaN
X_trunc = X[~np.isnan(X).any(axis=1)]
Y_trunc = Y[~np.isnan(X).any(axis=1)]
Y_trunc = Y_trunc.reshape(Y_trunc.shape[0])
print 'X_trunc.shape: ', X_trunc.shape
print 'Y_trunc.shape: ', Y_trunc.shape

X_train, X_test, Y_train, Y_test = train_test_split(X_trunc, Y_trunc, test_size=0.1, random_state=42)

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(C=10000.0))
])
clf.fit(X_train, Y_train)
preds = clf.predict(X_test)

score = metrics.f1_score(Y_test, preds)
print("f1-score:   %0.3f" % score)
