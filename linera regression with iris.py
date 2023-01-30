#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
iris1 = load_iris()


# In[55]:


iris= pd.read_csv("C:/Users/DITU.DESKTOP-KNV4SBH/Desktop/Iris.csv")


# In[56]:


print(iris)


# In[57]:


snsdata = iris.drop(['Id'], axis=1)
g = sns.pairplot(snsdata, hue='Species', markers='x')
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.kdeplot)


# In[58]:


X_train, X_test,Y_train,Y_test=train_test_split(iris1.data, iris1.target, random_state=1)
print(X_train.shape)
print(X_test.shape)


# In[59]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[60]:


scaler.fit(X_train)


# In[61]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_test, Y_test)


# In[62]:


pred=model.predict(X_test)
print("Accuracy of Model::",model.score(X_test, Y_test))


# In[63]:


print(mean_squared_error(Y_test, pred))

