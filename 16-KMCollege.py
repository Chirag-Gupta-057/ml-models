# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:23:10 2019

@author: Chirag
"""

import pandas as pd
import numpy as np

dataset=pd.read_csv('E:/College_Data.csv',index_col=0)



from sklearn.cluster import KMeans

model=KMeans(n_clusters=2)

labels=model.fit(dataset.drop(['Private'],axis=1))

labels=model.labels_

print('Cluster centroid ',model.cluster_centers_)

print(labels)

def conv(priv):
    if priv=='Yes':
        return 1
    else:
        return 0

dataset['Cluster']=dataset['Private'].apply(conv)

from sklearn.metrics import confusion_matrix

print('Accuracy:',confusion_matrix(labels,dataset['Cluster']))





