# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:25:11 2019

@author: Chirag
"""


import pandas as pd
import numpy as np
import seaborn as sns

dataset = pd.read_csv('E:/Classified Data.csv')
'''
pr=pd.get_dummies(dataset['purpose'],drop_first=True)

dataset=pd.concat([pr,dataset],axis=1)

dataset.drop(['purpose'],axis=1,inplace=True)'''

x=dataset.drop(['TARGET CLASS'],axis=1)
y=dataset['TARGET CLASS']


from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=0,test_size=0.2)

from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

print("Accuracy: ",model.score(xtest,ytest))

from sklearn.metrics import confusion_matrix
print('conf: ',confusion_matrix(ytest,ypred))