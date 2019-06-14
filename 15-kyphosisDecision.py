# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:39:37 2019

@author: Chirag
"""


import pandas as pd
import numpy as np
import seaborn as sns

dataset = pd.read_csv('E:/kyphosis.csv')

pr=pd.get_dummies(dataset['purpose'],drop_first=True)

dataset=pd.concat([pr,dataset],axis=1)

dataset.drop(['purpose'],axis=1,inplace=True)