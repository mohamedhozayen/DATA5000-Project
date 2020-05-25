# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 01:27:46 2019

 features_optimal.py                      
 SYSC 5405 - Term Project     
 Group 7: Decision Trees      
 By: Mo, Jason, Ian and Ben 

@author: Mohamed Hozayen
"""

import numpy as np
import pandas as pd
import preprocessing as prc
import feature_selection as fs
import model_evaluation
import matplotlib.pyplot as plt
import seaborn as sns

# optional: use to vote for best depth - insanity!!
def main_best_n(data, n): 
    l=[]
    for i in range(0,n):
        rslt = main.test_tree_depth(data)
        l.append([rslt.index(max(rslt)), max(rslt)])
    return l

def evaluate_corr(data, data_str_name, n_features, max_dapth):
    """
    Spearman Pearson ANOVA RFECV (ignore RFE)
    """
    summary_balance = []
    
#    pearson = fs.corr_linear(data, method='pearson')
#    spearman = fs.corr_linear(data, method='spearman')
#    p_class = pearson['class']
#    s_class = spearman['class']
#    
    for i in range(1, n_features+1):

#        pearson = pearson.drop(['class'], axis=1)
#        pearson = pd.concat([pearson.iloc[:,0:i], p_class], axis=1)
#        out = model_evaluation.test_tree_depth(pearson, class_weight="balanced")
#        summary_balance.append([data_str_name + '-pearson' , i, out.index(max(out)), max(out)])
#        
#        spearman = spearman.drop(['class'], axis=1)
#        spearman = pd.concat([spearman.iloc[:,0:i], s_class], axis=1)
#        out = model_evaluation.test_tree_depth(spearman, class_weight="balanced")
#        summary_balance.append([data_str_name + '-spearman', i, out.index(max(out)), max(out)])
        
        df = fs.select_k_best_ANOVA(data, k=n_features)
        out = model_evaluation.test_tree_depth(df, class_weight="balanced")
        summary_balance.append([data_str_name + '-ANOVA', i, out.index(max(out)), max(out)])
        
        df = fs.RFECV_DT(data, min_features_to_select=n_features, max_depth=max_dapth)
        out = model_evaluation.test_tree_depth(df, class_weight="balanced")
        summary_balance.append([data_str_name + '-RFECV', i,  out.index(max(out)), max(out)])

    return summary_balance

df = pd.read_csv('wdbc.data', sep=',', header=None)
header = ['id', 'class']
for i in range(1, 31):
    header.append('feature-'+str(i))
df.columns = header
df = df.drop(['id'], axis=1).replace(['M', 'B'], [1, 0])
features = df.drop(['class'], axis=1)
target = df['class']
df = pd.concat([features, target], axis=1)
df = prc.detect_outlier_iterative_IQR(df)
df = prc.handle_outlier(df)
df = prc.standarize(df) # or normalize

summary_balance = []
class_weight="balanced"

vt = fs.variance_threshold(df, 1)
rslt_vt = model_evaluation.test_tree_depth(vt, class_weight)
summary_balance.append(['variance-threshold', rslt_vt.index(max(rslt_vt)), max(rslt_vt)])

pca = fs.pca_linear(df, n=2) # n_c9 is 9, based VarianceThreshold results, axis to gain most information
rslt_pca = model_evaluation.test_tree_depth(pca, class_weight)
summary_balance.append(['pca-2', rslt_pca.index(max(rslt_pca)), max(rslt_pca)])

pca = fs.pca_linear(df, n=7) # n_c9 is 9, based VarianceThreshold results, axis to gain most information
rslt_pca = model_evaluation.test_tree_depth(pca, class_weight)
summary_balance.append(['pca-7', rslt_pca.index(max(rslt_pca)), max(rslt_pca)])

pca = fs.pca_linear(df, n=9) # n_c9 is 9, based VarianceThreshold results, axis to gain most information
rslt_pca = model_evaluation.test_tree_depth(pca, class_weight)
summary_balance.append(['pca-9', rslt_pca.index(max(rslt_pca)), max(rslt_pca)])

pca = fs.pca_linear(df, n=10) # n_c9 is 9, based VarianceThreshold results, axis to gain most information
rslt_pca = model_evaluation.test_tree_depth(pca, class_weight)
summary_balance.append(['pca-10', rslt_pca.index(max(rslt_pca)), max(rslt_pca)])

pca = fs.pca_linear(df, n=15) # n_c9 is 9, based VarianceThreshold results, axis to gain most information
rslt_pca = model_evaluation.test_tree_depth(pca, class_weight)
summary_balance.append(['pca-15', rslt_pca.index(max(rslt_pca)), max(rslt_pca)])

tsne_1 = fs.tsne(df, n=1) #3 is max
rslt_tsne = model_evaluation.test_tree_depth(tsne_1, class_weight)
summary_balance.append(['tsne-1', rslt_tsne.index(max(rslt_tsne)), max(rslt_tsne)])   
#tsne.to_csv('tsne-1.csv') 

tsne = fs.tsne(df, n=2) #3 is max
rslt_tsne = model_evaluation.test_tree_depth(tsne, class_weight)
summary_balance.append(['tsne-2', rslt_tsne.index(max(rslt_tsne)), max(rslt_tsne)])  

tsne = fs.tsne(df, n=3) #3 is max
rslt_tsne = model_evaluation.test_tree_depth(tsne, class_weight)
summary_balance.append(['tsne-3', rslt_tsne.index(max(rslt_tsne)), max(rslt_tsne)])  

pca_kernel = fs.pca_kernel(df, kernel='rbf') #kernel : “linear” | “poly” | “rbf” | “sigmoid” | “cosine” | “precomputed”
rslt_kernel = model_evaluation.test_tree_depth(pca_kernel, class_weight)
summary_balance.append(['rbf-', rslt_kernel.index(max(rslt_kernel)), max(rslt_kernel)])

pca_kernel = fs.pca_kernel(df, kernel='poly') 
rslt_kernel = model_evaluation.test_tree_depth(pca_kernel, class_weight)
summary_balance.append(['poly-', rslt_kernel.index(max(rslt_kernel)), max(rslt_kernel)])

pca_cos = fs.pca_kernel(df, kernel='cosine')
rslt_kernel = model_evaluation.test_tree_depth(pca_cos, class_weight)
summary_balance.append(['cosine-', rslt_kernel.index(max(rslt_kernel)), max(rslt_kernel)])
#pca_kernel.to_csv('pca-cosine.csv') 

summary_table_balance = pd.DataFrame(summary_balance)
summary_table_balance.columns = ['method-balance', 'optimal-depth', 'pre@recall50']
summary_table_balance = summary_table_balance.sort_values(by=['pre@recall50'], ascending = False)
summary_table_balance.to_csv('unsupervised-features-balance-summary_table.csv') 


#pca_cos = pd.read_csv('pca-cosine.csv', sep=',').drop(['index'], axis=1) 
#tsne1 = pd.read_csv('tsne-1.csv', sep=',').drop(['index'], axis=1) 


summary_balance_supervised = []
summary_balance_supervised.extend(evaluate_corr(pca_cos, 'pca_cos', n_features=10, max_dapth=9))
#summary_balance_supervised.extend(evaluate_corr(tsne1, 'tsne_1', n_features=10, max_dapth=9))

summary_table_balance_supervised = pd.DataFrame(summary_balance_supervised)
summary_table_balance_supervised.columns = ['method-balance', 'n_features', 'optimal-depth', 'pre@recall50']
tsne1_data = ['tsne-1', 1, 3, 0.8803418803418803]
tsne2_data = ['tsne-2', 2, 6, 0.8969072164948454]
tsne3_data = ['tsne-3', 3, 3, 0.7545454545454545]
summary_table_balance_supervised.loc[len(summary_table_balance_supervised)] = tsne1_data
summary_table_balance_supervised.loc[len(summary_table_balance_supervised)] = tsne2_data
summary_table_balance_supervised.loc[len(summary_table_balance_supervised)] = tsne3_data
summary_table_balance_supervised = summary_table_balance_supervised.sort_values(by=['pre@recall50'], ascending = False)
#summary_table_balance_supervised.to_csv('supervised-features-balance-summary_table.csv')

#summary_table_balance_supervised[summary_table_balance_supervised['method-balance']=='pca_cos-ANOVA'] 

pca_cos = fs.pca_kernel(df, kernel='cosine')
optimal_features = fs.select_k_best_ANOVA(pca_cos, k=7)
#optimal_features.to_csv('optimal_features_pca_cos.csv') 

t1 = pd.read_csv('supervised-features-balance-summary_table.csv', sep=',')
t2 = pd.read_csv('unsupervised-features-balance-summary_table.csv', sep=',')


"""
summary_balance = []

summary_table_balance = pd.DataFrame(summary_balance)
summary_table_balance.columns = ['method-balance', 'n_features', 'optimal-depth', 'pre@recall50']
summary_table_balance = summary_table_balance.sort_values(by=['pre@recall50'], ascending = False)
summary_table_balance.to_csv('supervised-features-balance-summary_table.csv') 

depth_2 = summary_table_balance[summary_table_balance['optimal-depth']==2]
depth_3 = summary_table_balance[summary_table_balance['optimal-depth']==3]
depth_4 = summary_table_balance[summary_table_balance['optimal-depth']==4]
depth_5 = summary_table_balance[summary_table_balance['optimal-depth']==5]
depth_6 = summary_table_balance[summary_table_balance['optimal-depth']==6]
depth_7 = summary_table_balance[summary_table_balance['optimal-depth']==7]

summary_balance.extend(evaluate_corr(pca_rbf, 'pca_rbf', n_features=10, max_dapth=7))
summary_balance.extend(evaluate_corr(pca_cos, 'pca_cos', n_features=10, max_dapth=7))
summary_balance.extend(evaluate_corr(pca_poly, 'pca_poly', n_features=10, max_dapth=7))
summary_balance.extend(evaluate_corr(pca_10, 'pca_10', n_features=10, max_dapth=7))


#summary = []
#summary_balance = []

#summary_table_balance = pd.DataFrame(summary_balance)
#summary_table_balance.columns = ['method-balance', 'optimal tree depth', 'pre@recall50']
#summary_table_balance.to_csv('unsupervised-features-balance-summary_table.csv')

pca_rbf = fs.pca_kernel(df, kernel='rbf')
pca_poly = fs.pca_kernel(df, kernel='poly') 
pca_cos = fs.pca_kernel(df, kernel='cosine')
pca_10= fs.pca_linear(df, n=10)

pca_10.to_csv('pca-10-features.csv')
pca_rbf.to_csv('pca-rbf-features.csv')
pca_poly.to_csv('pca-poly-features.csv')
pca_cos.to_csv('pca-cos-features.csv')


vt = fs.variance_threshold(df, threshold=1)
rslt_vt = main.test_tree_depth(vt, class_weight="balanced")
summary_balance.append(['variance-threshold', rslt_vt.index(max(rslt_vt)), max(rslt_vt)])

pca = fs.pca_linear(df, n=2) # n_c9 is 9, based VarianceThreshold results, axis to gain most information
rslt_pca = main.test_tree_depth(pca, class_weight="balanced")
summary_balance.append(['pca-2', rslt_pca.index(max(rslt_pca)), max(rslt_pca)])

pca = fs.pca_linear(df, n=7) # n_c9 is 9, based VarianceThreshold results, axis to gain most information
rslt_pca = main.test_tree_depth(pca, class_weight="balanced")
summary_balance.append(['pca-7', rslt_pca.index(max(rslt_pca)), max(rslt_pca)])

pca = fs.pca_linear(df, n=9) # n_c9 is 9, based VarianceThreshold results, axis to gain most information
rslt_pca = main.test_tree_depth(pca, class_weight="balanced")
summary_balance.append(['pca-9', rslt_pca.index(max(rslt_pca)), max(rslt_pca)])

pca = fs.pca_linear(df, n=10) # n_c9 is 9, based VarianceThreshold results, axis to gain most information
rslt_pca = main.test_tree_depth(pca, class_weight="balanced")
summary_balance.append(['pca-10', rslt_pca.index(max(rslt_pca)), max(rslt_pca)])

pca = fs.pca_linear(df, n=15) # n_c9 is 9, based VarianceThreshold results, axis to gain most information
rslt_pca = main.test_tree_depth(pca, class_weight="balanced")
summary_balance.append(['pca-15', rslt_pca.index(max(rslt_pca)), max(rslt_pca)])

tsne = fs.tsne(df, n=1) #3 is max
rslt_tsne = main.test_tree_depth(tsne, class_weight="balanced")
summary_balance.append(['tsne-1', rslt_tsne.index(max(rslt_tsne)), max(rslt_tsne)])   

tsne = fs.tsne(df, n=2) #3 is max
rslt_tsne = main.test_tree_depth(tsne, class_weight="balanced")
summary_balance.append(['tsne-2', rslt_tsne.index(max(rslt_tsne)), max(rslt_tsne)])  

tsne = fs.tsne(df, n=3) #3 is max
rslt_tsne = main.test_tree_depth(tsne, class_weight="balanced")
summary_balance.append(['tsne-3', rslt_tsne.index(max(rslt_tsne)), max(rslt_tsne)])  

pca_kernel = fs.pca_kernel(df, kernel='rbf') #kernel : “linear” | “poly” | “rbf” | “sigmoid” | “cosine” | “precomputed”
rslt_kernel = main.test_tree_depth(pca_kernel, class_weight="balanced")
summary_balance.append(['rbf-', rslt_kernel.index(max(rslt_kernel)), max(rslt_kernel)])

pca_kernel = fs.pca_kernel(df, kernel='poly') 
rslt_kernel = main.test_tree_depth(pca_kernel, class_weight="balanced")
summary_balance.append(['poly-', rslt_kernel.index(max(rslt_kernel)), max(rslt_kernel)])

pca_kernel = fs.pca_kernel(df, kernel='cosine')
rslt_kernel = main.test_tree_depth(pca_kernel, class_weight="balanced")
summary_balance.append(['cosine-', rslt_kernel.index(max(rslt_kernel)), max(rslt_kernel)])


depth_2 = summary_table_balance[summary_table_balance['optimal-depth']==2]
depth_3 = summary_table_balance[summary_table_balance['optimal-depth']==3]
depth_4 = summary_table_balance[summary_table_balance['optimal-depth']==4]
depth_5 = summary_table_balance[summary_table_balance['optimal-depth']==5]
depth_6 = summary_table_balance[summary_table_balance['optimal-depth']==6]
depth_7 = summary_table_balance[summary_table_balance['optimal-depth']==7]

depth_2.to_csv('depth-2.csv') 
depth_3.to_csv('depth-3.csv') 
depth_4.to_csv('depth-4.csv') 
depth_5.to_csv('depth-5.csv') 
depth_6.to_csv('depth-6.csv') 
depth_7.to_csv('depth-7.csv') 

summary_tree = pd.DataFrame()

temp = depth_2[depth_2['pre@recall50'] > 0.075]
summary_tree = summary_tree.append(temp)

temp = depth_3[depth_3['pre@recall50'] > 0.076]
summary_tree = summary_tree.append(temp)

temp = depth_4[depth_4['pre@recall50'] > 0.076]
summary_tree = summary_tree.append(temp)

temp = depth_5[depth_5['pre@recall50'] > 0.076]
summary_tree = summary_tree.append(temp)

temp = depth_6[depth_6['pre@recall50'] > 0.076]
summary_tree = summary_tree.append(temp)

temp = depth_7[depth_7['pre@recall50'] > 0.076]
summary_tree = summary_tree.append(temp)


summary_tree.to_csv('performance-optimal-summary-trees.csv') 


plt.figure(figsize=(20,20))
plt.tight_layout()
ax1 = plt.subplot(231)
ax1.title.set_text('Performance: optimal depth of 2')
ax1.legend(loc="best")
sns.scatterplot(
            x="n_features", y="pre@recall50",
            hue="method-balance",
            data=depth_2,
            legend="full",
            ax=ax1
)

ax2 = plt.subplot(232)
ax2.title.set_text('Performance: optimal depth of 3')
ax2.legend(loc="best")
sns.scatterplot(
            x="n_features", y="pre@recall50",
            hue="method-balance",
            data=depth_3,
            legend="full",
            ax=ax2
)

ax3 = plt.subplot(233)
ax3.title.set_text('Performance: optimal depth of 4')
ax3.legend(loc="best")
sns.scatterplot(
            x="n_features", y="pre@recall50",
            hue="method-balance",
            data=depth_4,
            legend="full",
            ax=ax3
)

ax4 = plt.subplot(234)
ax4.title.set_text('Performance: optimal depth of 5')
ax4.legend(loc="best")
sns.scatterplot(
            x="n_features", y="pre@recall50",
            hue="method-balance",
            data=depth_5,
            legend="full",
            ax=ax4
)

ax5 = plt.subplot(235)
ax5.title.set_text('Performance: optimal depth of 6')
ax5.legend(loc="best")
sns.scatterplot(
            x="n_features", y="pre@recall50",
            hue="method-balance",
            data=depth_6,
            legend="full",
            ax=ax5
)

ax6 = plt.subplot(236)
ax6.title.set_text('Performance: optimal depth of 7')
ax6.legend(loc="best")
sns.scatterplot(
            x="n_features", y="pre@recall50",
            hue="method-balance",
            data=depth_7,
            legend="full",
            ax=ax6
)

plt.savefig('Files\supervised-features-balance-summary_plot.pdf')


# Print out the distribution of curves 
#plt.plot()
#plt.ylabel("Depth of Tree")
#plt.xlabel("Pr@Re>50")
#plt.title("Testing Decision Tree Depth")
#plt.xticks(list(range(2, len(rslt))))
#plt.show()


plt.figure(figsize=(25,17))
ax1 = plt.subplot(231)
ax1.title.set_text('Performance: optimal depth of 2')
ax1.legend(loc="best")
sns.scatterplot(
            x="n_features", y="pre@recall50",
            hue="method-balance",
            data=depth_2,
            legend="full",
            ax=ax1
)

plt.savefig('Files\supervised-features-balance-DT-2.png')


plt.figure(figsize=(25,17))
ax2 = plt.subplot(232)
ax2.title.set_text('Performance: optimal depth of 3')
ax2.legend(loc="best")
sns.scatterplot(
            x="n_features", y="pre@recall50",
            hue="method-balance",
            data=depth_3,
            legend="full",
            ax=ax2
)
plt.savefig('Files\supervised-features-balance-DT-3.png')

plt.figure(figsize=(25,17))
ax3 = plt.subplot(233)
ax3.title.set_text('Performance: optimal depth of 4')
ax3.legend(loc="best")
sns.scatterplot(
            x="n_features", y="pre@recall50",
            hue="method-balance",
            data=depth_4,
            legend="full",
            ax=ax3
)
plt.savefig('Files\supervised-features-balance-DT-4.png')

plt.figure(figsize=(25,17))
ax4 = plt.subplot(234)
ax4.title.set_text('Performance: optimal depth of 5')
ax4.legend(loc="best")
sns.scatterplot(
            x="n_features", y="pre@recall50",
            hue="method-balance",
            data=depth_5,
            legend="full",
            ax=ax4
)
plt.savefig('Files\supervised-features-balance-DT-5.png')

plt.figure(figsize=(25,17))
ax5 = plt.subplot(235)
ax5.title.set_text('Performance: optimal depth of 6')
ax5.legend(loc="best")
sns.scatterplot(
            x="n_features", y="pre@recall50",
            hue="method-balance",
            data=depth_6,
            legend="full",
            ax=ax5
)
plt.savefig('Files\supervised-features-balance-DT-6.png')


plt.figure(figsize=(25,17))
ax6 = plt.subplot(236)
ax6.title.set_text('Performance: optimal depth of 7')
ax6.legend(loc="best")
sns.scatterplot(
            x="n_features", y="pre@recall50",
            hue="method-balance",
            data=depth_7,
            legend="full",
            ax=ax6
)
plt.savefig('Files\supervised-features-balance-DT-7.png')

"""
