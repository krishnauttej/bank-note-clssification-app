# -*- coding: utf-8 -*-
"""
Created on Sun May 17 18:10:11 2020

@author: Home
"""

import pandas as pd
import numpy as np
df=pd.read_csv('BankNote_Authentication.csv')
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)
import pickle 
pickle.dump(classifier, open("classifier.pkl","wb"))