# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 16:56:34 2019

@author: Chirag
"""


import pandas as pd
import numpy as np
import seaborn as sns

dataset = pd.read_csv('E:/loan_data.csv')

pr=pd.get_dummies(dataset['purpose'],drop_first=True)

dataset=pd.concat([pr,dataset],axis=1)

dataset.drop(['purpose'],axis=1,inplace=True)

x=dataset.drop(['not.fully.paid'],axis=1)
y=dataset['not.fully.paid']


from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=0,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
mod=RandomForestClassifier(n_estimators=84)

mod.fit(xtrain,ytrain)

ypred=mod.predict(xtest)

print('Accu: ',mod.score(xtest,ytest))



import matplotlib.pyplot as plt
e=[]
for i in range(1,101):
    rnn=RandomForestClassifier(n_estimators=i)
    rnn.fit(xtrain,ytrain)
    pre_i=rnn.predict(xtest)
    e.append(np.mean(pre_i!=ytest))
    
plt.plot(range(1,101),e,marker='o',ls='--',markerfacecolor='r')
plt.show()
