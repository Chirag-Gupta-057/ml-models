# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:49:52 2019

@author: Chirag
"""


import pandas as pd
import numpy as np
import seaborn as sns

dataset = pd.read_csv('E:/titanic_train.csv')

def conv(col):
    age=col[0]
    pclass=col[1]
    if pd.isnull(age):
        if pclass==1:
            return 37
        elif pclass==2:
            return 29
        else:
            return 24
    return age

dataset['Age']=dataset[['Age','Pclass']].apply(conv,axis=1)

embark=pd.get_dummies(dataset['Embarked'],drop_first=True)
sex=pd.get_dummies(dataset['Sex'],drop_first=True)

dataset=pd.concat([embark,dataset,sex],axis=1)

dataset.drop(['Cabin','PassengerId','Sex','Name','Embarked','Ticket'],axis=1,inplace=True)

x=dataset.drop(['Survived'],axis=1)
y=dataset['Survived']

from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=0,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=23)

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