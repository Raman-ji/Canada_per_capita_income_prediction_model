#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model


# In[12]:


df=pd.read_csv(r"C:\Users\uniqu\Downloads\archive\canada_per_capita_income.csv")
df


# In[5]:


plt.scatter(df.year, df['per capita income (US$)'])


# In[7]:


plt.xlabel('Year')
plt.ylabel('income')
plt.scatter(df.year,df['per capita income (US$)'], marker='+')


# In[9]:


reg= linear_model.LinearRegression()
reg.fit(df[['year']], df['per capita income (US$)'])


# In[18]:


reg.predict([[2025]])


# In[ ]:




