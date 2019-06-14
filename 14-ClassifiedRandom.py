# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:29:44 2019

@author: Chirag
"""


import pandas as pd
import numpy as np
import seaborn as sns

dataset = pd.read_csv('E:/Classified Data.csv')


x=dataset.drop(['TARGET CLASS'],axis=1)
y=dataset['TARGET CLASS']


from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=0,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=35)
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

print("Accuracy: ",model.score(xtest,ytest))



import matplotlib.pyplot as plt
e=[]
for i in range(1,101):
    rnn=RandomForestClassifier(n_estimators=i)
    rnn.fit(xtrain,ytrain)
    pre_i=rnn.predict(xtest)
    e.append(np.mean(pre_i!=ytest))
    
plt.plot(range(1,101),e,marker='o',ls='--',markerfacecolor='r')
plt.show()

