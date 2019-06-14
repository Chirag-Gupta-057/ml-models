# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:17:36 2019

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

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=20)

model = model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

print('Accuracy: ',model.score(xtest,ytest))


import matplotlib.pyplot as plt
e=[]
for i in range(1,51):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(xtrain,ytrain)
    pre_i=knn.predict(xtest)
    e.append(np.mean(pre_i!=ytest))
    
plt.plot(range(1,51),e,marker='o',ls='--',markerfacecolor='r')
plt.show()
