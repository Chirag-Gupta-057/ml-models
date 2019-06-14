# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
dataset = pd.read_csv('E:/Salary_Data.csv')
x=dataset.drop('Salary',axis=1)
y=dataset['Salary']
from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=0,test_size=0.2)

from sklearn.linear_model import LinearRegression

model=LinearRegression()

model = model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

print('Coefficient: ',model.intercept_)

print('Accuracy: ',model.score(xtest,ytest))
