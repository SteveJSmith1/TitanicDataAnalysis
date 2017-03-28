#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 10:11:14 2017

@author: steve
"""

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Reading Data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

# Describing data

print(train_df.columns.values)
"""
['PassengerId' 'Survived' 'Pclass' ..., 'Fare' 'Cabin' 'Embarked']
"""

print(test_df.columns.values)
"""
['PassengerId' 'Pclass' 'Name' ..., 'Fare' 'Cabin' 'Embarked']
"""

# Categorical data:
    
# Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.

# Numerical data:
    
# Continuous: Age, Fare. Discrete: SibSp, Parch.

