#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd
df = pd.read_csv('liver.csv')


# In[3]:


df


# In[4]:


train_data = df.drop(['Selector'],axis='columns')
train_label = pd.DataFrame(df['Selector'])


# In[5]:


#split the data into 70% training and 30% testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data,train_label,test_size=0.3)


# In[6]:


X_train


# In[7]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


# In[8]:


from sklearn.metrics import mean_squared_error


# In[9]:


mean_squared_error(y_test,clf.predict(X_test))


# In[10]:


from sklearn.naive_bayes import GaussianNB


# In[11]:


gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb.score(X_test,y_test)


# In[12]:


mean_squared_error(y_test,gnb.predict(X_test))


# In[13]:


import matplotlib.pyplot as plt


# In[14]:


plt.scatter(X_test.iloc[:,0], y_test.iloc[:,0], color='black')
plt.scatter(X_test.iloc[:,0], clf.predict(X_test))


# In[15]:


dfop = pd.DataFrame(df['GOT']/df['GPT'],columns=['GOT/GPT'])


# In[16]:


dfop


# In[17]:


dfpo = pd.DataFrame(df['GPT']/df['GOT'],columns=['GPT/GOT'])


# In[18]:


dfpo


# In[19]:


df2=pd.concat([dfpo,dfop,df['GOT'],df['GPT'],df['Selector']],axis=1)


# In[20]:


df2


# In[21]:


correlation_matrix = df2.corr()


# In[22]:


import seaborn as sns
ax = sns.heatmap(correlation_matrix, annot=True)


# In[23]:


df3=pd.concat([dfpo,dfop,df],axis=1)
df3=df3.drop(['GOT','GPT'],axis='columns')


# In[24]:


df3


# In[25]:


train_data = df3.drop(['Selector'],axis='columns')
train_label = pd.DataFrame(df3['Selector'])


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(train_data,train_label,test_size=0.3)


# In[27]:


X_train


# In[28]:


clf = LogisticRegression()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


# In[29]:


mean_squared_error(y_test,clf.predict(X_test))


# In[30]:


plt.scatter(X_test.iloc[:,0], y_test.iloc[:,0], color='black')
plt.scatter(X_test.iloc[:,0], clf.predict(X_test))


# In[31]:


gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb.score(X_test,y_test)


# In[32]:


mean_squared_error(y_test,gnb.predict(X_test))


# In[33]:


plt.scatter(X_test.iloc[:,0], y_test.iloc[:,0], color='black')
plt.scatter(X_test.iloc[:,0], gnb.predict(X_test))

