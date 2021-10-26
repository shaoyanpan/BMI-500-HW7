# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 01:35:46 2021

@author: hi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 00:53:09 2021

@author: hi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 23:37:54 2021

@author: hi
"""

import sqlite3
import pandas as pd 
from glob import glob
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

#Read your csv file and shuffle it
df = pd.read_csv('C:/PAN/BMI 500/HW9/third_ten.csv')
df = df.sample(frac=1).reset_index(drop=True)
columns = df.columns
labels = df['aki_stage'].values
onehot_labels = pd.get_dummies(labels).values

#Interpolate the missing values and drop the aki stage (label)
df = df.astype(float)
df = df.interpolate(method='linear', axis=0).ffill().bfill()
df.drop(['aki_stage'],axis=1,inplace=True)

# Start the Kmeans and show the result
mat = df.values
km = KMeans(n_clusters=4)
km.fit(mat)
result = km.fit_predict(mat)
cao = labels==result
print('accuracy = '+str(sum(labels==result)/len(result)*100)+'%')
print('Stage 0 accuracy = '+str(sum(cao[labels==0])/len(labels==0)*100)+'%')
print('Stage 1 accuracy = '+str(sum(cao[labels==1])/len(labels==1)*100)+'%')
print('Stage 2 accuracy = '+str(sum(cao[labels==2])/len(labels==2)*100)+'%')
print('Stage 3 accuracy = '+str(sum(cao[labels==3])/len(labels==3)*100)+'%')


# Visualize the data by the ground truth
new_mat = df
X_embedded  = TSNE(n_components=2, learning_rate='auto',
                 init='random', n_iter=250).fit_transform(df)

df_subset = pd.DataFrame([])
df_subset['tsne-2d-one'] = X_embedded[:,0]
df_subset['tsne-2d-two'] = X_embedded[:,1]
df_subset['y'] = result
# plt.scatter(X_embedded[:,0], X_embedded[:,1])
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", len(df_subset['y'].unique())),
    data=df_subset,
    legend="full",
    alpha=0.3
)

df_subset = pd.DataFrame([])
df_subset['tsne-2d-one'] = X_embedded[:,0]
df_subset['tsne-2d-two'] = X_embedded[:,1]
df_subset['y'] = labels
# plt.scatter(X_embedded[:,0], X_embedded[:,1])
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", len(df_subset['y'].unique())),
    data=df_subset,
    legend="full",
    alpha=0.3
)



#MLP for classification
X_train = df.iloc[:round(len(df)*0.8)]
y_train = onehot_labels[:round(len(df)*0.8)]
X_test = df.iloc[round(len(df)*0.8)+1:]
y_test = onehot_labels[round(len(df)*0.8)+1:]
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)
X_test = df.iloc[round(len(df)*0.8)+1:]
y_test = onehot_labels[round(len(df)*0.8)+1:]
max_value = X_train.max()
min_value = X_train.min()
X_train -= (max_value+min_value)/2
X_train /= (max_value-min_value)/2
X_test -= (max_value+min_value)/2
X_test /= (max_value-min_value)/2
clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100,alpha=0.001).fit(X_train, y_train)
final_result = clf.predict(X_test)

cao = y_test.argmax(1) ==final_result.argmax(1)
print('accuracy ='+str( np.sum(y_test.argmax(1) ==final_result.argmax(1))/len(y_test)))
print('Stage 0 accuracy = '+str(sum(cao[y_test.argmax(1)==0])/sum(y_test.argmax(1)==0)*100)+'%')
print('Stage 1 accuracy = '+str(sum(cao[y_test.argmax(1)==1])/sum(y_test.argmax(1)==1)*100)+'%')
print('Stage 2 accuracy = '+str(sum(cao[y_test.argmax(1)==2])/sum(y_test.argmax(1)==2)*100)+'%')
print('Stage 3 accuracy = '+str(sum(cao[y_test.argmax(1)==3])/sum(y_test.argmax(1)==3)*100)+'%')
