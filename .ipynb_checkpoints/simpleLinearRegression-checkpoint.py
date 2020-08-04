# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:14:42 2020

@author: Faraz Ahmad Khan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet = pd.read_csv('Salary_Data.csv')
X = dataSet.iloc[:,:-1].values
Y = dataSet.iloc[:,1].values
 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 1/3, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
Y_train=Y_train.reshape(-1,1)

regressor.fit(X_train,Y_train)
X_test = X_test.reshape(-1,1)
y_pred = regressor.predict(X_test)

plt.scatter(X_train, Y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title('Salary Vs Experence (Training set)')
plt.ylabel('Year of Experence')
plt.xlabel('salary')
plt.show()

plt.scatter(X_test, Y_test,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title('Salary Vs Experence ( Test set)')
plt.ylabel('Year of Experence')
plt.xlabel('salary')
plt.show()