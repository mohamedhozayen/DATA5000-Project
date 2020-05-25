# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:48:09 2020

@author: mohamedhozayen
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import * 

df = pd.read_csv('wdbc.data', sep=',', header=None)
header = ['id', 'class']
for i in range(1, 31):
    header.append('feature-'+str(i))
df.columns = header
df = df.drop(['id'], axis=1).replace(['M', 'B'], [1, 0])

df = optimal_features
# Split into train and test
X = df.drop(['class'], axis=1)
Y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4)

# Train and test model
dt = DecisionTreeClassifier(max_depth = 2, class_weight="balanced")
dt.fit(X_train, y_train) 
y_pred = dt.predict(X_test) 


# Print results 
print("Confusion Matrix: \n" + str(confusion_matrix(y_test, y_pred)))
print ("Accuracy : " + str(accuracy_score(y_test,y_pred)*100))
print("Report : \n" + str(classification_report(y_test, y_pred)))