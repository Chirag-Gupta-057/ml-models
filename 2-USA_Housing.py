# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np
dataset = pd.read_csv(filepath_or_buffer = "C:/Users/Chirag/.spyder-py3/USA_Housing.csv")

y=dataset['Price']
x=dataset.drop(['Address','Price'],axis=1)

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