
# Imports
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import preprocessing as prc
import feature_selection as fs
from random import seed
from random import randint
from sklearn.model_selection import *
from sklearn.tree import * 
from sklearn.metrics import * 
from sklearn.feature_selection import *
from sklearn.ensemble import AdaBoostClassifier
from datetime import datetime
################################
startTime = datetime.now()
bootstrap_test_count = 10
rand_state = randint(0, 100)
################################

# Report will plot a PR curve and return the test stat
def report(name, y_true, y_pred, y_prob, verbos=False):

    cm = confusion_matrix(y_true, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    average_precision = average_precision_score(y_true, y_prob)
    PrecisionAtRe50_DT = np.max(precision[recall>0.5])
    
    if verbos:
        print("===== " + name +"=====")
        print("TN = " + str(cm[0][0]))
        print("FP = " + str(cm[0][1]))
        print("FN = " + str(cm[1][0]))
        print("TP = " + str(cm[1][1]))
        print("Pr = " + str(cm[1][1]/(cm[1][1] + cm[0][1])))
        print("Re = " + str(cm[1][1]/(cm[1][1] + cm[1][0])))
        print("Confusion Matrix: \n" + str(cm))
        print('Pr@Re50 = ', PrecisionAtRe50_DT)
        print()
        plt.plot(recall, precision,label=name + ", Pr@Re>50 = {0:.5f}".format(PrecisionAtRe50_DT))
   
    return PrecisionAtRe50_DT

# This function trains and tests a model
# Returns the predictions and their probabilty 
def test_model(model, X_train, X_test, y_train):
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	y_prob = model.predict_proba(X_test)[::,1]
	return y_pred, y_prob

# Main function
# Inputs: model - will be trained and validated using k-fold
# [optional] clean_data or unsupervise_fs
def main(df, name, model, unsupervise_fs = False):
    if 'id' in df:
        df = df.drop(['id'], axis=1)
    X = df.drop(['class'], axis=1)
    y = df['class']
	
    y_pred = []
    y_prob = []
    y_true = []
    kf = StratifiedKFold(n_splits=3, shuffle = True, random_state = rand_state)
    for train_index, test_index in kf.split(X, y):
		# Split train and test set
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#        print(len(y_test))
		# Train and test model  
        pred, prob = test_model(model, X_train, X_test, y_train)
        y_pred.extend(pred)
        y_prob.extend(prob)
        y_true.extend(y_test)

    return report(name, y_true, y_pred, y_prob)

# Evaluates the depth of the tree
def test_tree_depth(data, class_weight=None):
    test_stats = [0,0]
    for i in range(2, 10): # 2 to 15
        dt = DecisionTreeClassifier(max_depth = i, class_weight=class_weight)
        test_stats.append(main(df=data, name="DT with depth = "+str(i), model=dt))
    return test_stats





