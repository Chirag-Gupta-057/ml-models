# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 23:39:13 2019

@author: Chirag
"""

import pandas as pd
import numpy as np
import seaborn as sns

dataset = pd.read_csv('E:/titanic_train.csv')

sns.heatmap(dataset.isnull(),cbar = False,yticklabels =False,cmap = 'viridis')
sns.set_style('whitegrid')
sns.boxplot(y ='Age',x ='Pclass',data =dataset)

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
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=0,test_size=0.055)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model = model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

print('Accuracy: ',model.score(xtest,ytest))

from sklearn.metrics import confusion_matrix
print('Conf',confusion_matrix(ytest,ypred))




print('Intercept: ',model.intercept_)
print('Coefficent: ',model.coef_)

from sklearn.metrics import mean_absolute_error,mean_squared_error

print('MAE: ',mean_absolute_error(ytest,ypred))

print('MSE: ',mean_squared_error(ytest,ypred))

print('RMSE: ',np.sqrt(mean_squared_error(ytest,ypred)))

'''
y = dataset['Survived']
x = dataset.drop(['PassengerId','Name','Sex','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Survived'],axis=1)

from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=0,test_size=0.2)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model = model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

print('Intercept: ',model.intercept_)

print('Accuracy: ',model.score(xtest,ytest))

print('Coefficent: ',model.coef_)

from sklearn.metrics import mean_absolute_error,mean_squared_error

print('MAE: ',mean_absolute_error(ytest,ypred))

print('MSE: ',mean_squared_error(ytest,ypred))

print('RMSE: ',np.sqrt(mean_squared_error(ytest,ypred)))
'''
