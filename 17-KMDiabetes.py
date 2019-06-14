# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:50:58 2019

@author: Chirag
"""


import pandas as pd
import numpy as np

dataset=pd.read_csv('E:/diabetes.csv')



from sklearn.cluster import KMeans

model=KMeans(n_clusters=2)


labels=model.fit(dataset.drop(['Outcome'],axis=1))

labels=model.labels_

print('Cluster centroid ',model.cluster_centers_)

print(labels)


from sklearn.metrics import confusion_matrix

print('Accuracy:',confusion_matrix(labels,dataset['Outcome']))


