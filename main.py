# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:01:30 2020

@author: mohamedhozayen
"""

import numpy as np
import pandas as pd
import model_evaluation

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import * 

df = pd.read_csv('wdbc.data', sep=',', header=None)
header = ['id', 'class']
for i in range(1, 31):
    header.append('feature-'+str(i))
df.columns = header
df = df.drop(['id'], axis=1).replace(['M', 'B'], [1, 0])
features = df.drop(['class'], axis=1)
target = df['class']
df = pd.concat([features, target], axis=1)

model_evaluation.test_tree_depth(df, class_weight="balanced")