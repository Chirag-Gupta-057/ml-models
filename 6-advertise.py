# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 17:54:29 2019

@author: Chirag
"""


import pandas as pd
import numpy as np
import seaborn as sns

dataset = pd.read_csv('E:/advertising.csv')


dataset.drop(['Ad Topic Line','City','Country','Timestamp'],axis=1,inplace=True)


x=dataset.drop(['Clicked on Ad'],axis=1)
y=dataset['Clicked on Ad']

from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=0,test_size=0.2)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model = model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

print('Accuracy: ',model.score(xtest,ytest))