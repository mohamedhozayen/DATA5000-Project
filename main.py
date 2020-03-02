# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:01:30 2020

@author: mohamedhozayen
"""

import pandas as pd
import preprocessing as prc
import feature_selection as fs

df = pd.read_csv('wdbc-labelled.data', sep=',')

df = prc.detect_outlier_iterative_IQR(df)
df = prc.handle_outlier(df)
df = prc.standarize(df) # or normalize

pca_cos = fs.pca_kernel(df, kernel='cosine')
optimal_features = fs.select_k_best_ANOVA(pca_cos, k=7)


import time
start_time = time.time()

# =============================================================================
# df = pd.read_csv('wdbc.data', sep=',', header=None)
# 
# header = ['id', 'class']
# for i in range(1, 31):
#     header.append('feature-'+str(i))
# df.columns = header
# df = df.drop(['id'], axis=1).replace(['M', 'B'], [1, 0])
# features = df.drop(['class'], axis=1)
# target = df['class']
# df = pd.concat([features, target], axis=1)
# 
# df.to_csv('wdbc-labelled.data', index=False) 
# =============================================================================



print("--- %s seconds ---" % (time.time() - start_time))
