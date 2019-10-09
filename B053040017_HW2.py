#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('house-prices.csv')


# In[2]:


df


# In[3]:


#encode the string elements with number
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in df.columns[df.dtypes==object].to_list():
    df[i] = le.fit_transform(df[i].astype(str))


# In[4]:


#fill the NaN value with mean
from sklearn.impute import SimpleImputer
import numpy as np

ipt = SimpleImputer(missing_values=np.NaN, strategy='mean')
df = pd.DataFrame(ipt.fit_transform(df),columns=df.columns)


# In[5]:


#prepare the training data
train_data = df.drop(['SalePrice','Id'],axis='columns')
train_label = pd.DataFrame(df['SalePrice'])


# In[6]:


train_data


# In[7]:


#split the data into 70% training and 30% testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data,train_label,test_size=0.3)


# In[8]:


X_train


# In[9]:


from sklearn.metrics import mean_squared_error


# In[10]:


from sklearn.linear_model import LinearRegression
linearR_model = LinearRegression()
linearR_model.fit(X_train,y_train)
linear_result = linearR_model.predict(X_test)

print("linear regression model score:")
linearR_model.score(X_test,y_test)


# In[11]:


mean_squared_error(y_test,linear_result)


# In[12]:


from sklearn.linear_model import Ridge

clf = Ridge(alpha=1)
clf.fit(X_train,y_train)


# In[13]:


print("Ridge score:")
clf.score(X_test,y_test)


# In[14]:


mean_squared_error(y_test,clf.predict(X_test))


# In[15]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1)
lasso.fit(X_train, y_train)


# In[16]:


print("Lasso score:")
lasso.score(X_test,y_test)


# In[17]:


mean_squared_error(y_test,lasso.predict(X_test))


# In[18]:


import matplotlib.pyplot as plt

plt.scatter(X_test.iloc[:,0], y_test.iloc[:,0], color='black')


# In[19]:


plt.scatter(X_test.iloc[:,0],clf.predict(X_test))


# In[20]:


plt.scatter(X_test.iloc[:,0], y_test.iloc[:,0], color='black')
plt.scatter(X_test.iloc[:,0],clf.predict(X_test))


# In[21]:


from sklearn import preprocessing


# In[22]:


preprod_data = preprocessing.scale(train_data)
preprod_label = preprocessing.scale(train_label)
X_train, X_test, y_train, y_test = train_test_split(preprod_data,preprod_label,test_size=0.3)


# In[23]:


clf.fit(X_train,y_train)
clf.score(X_test,y_test)


# In[24]:


mean_squared_error(y_test,clf.predict(X_test))


# In[25]:


plt.scatter(X_test[:,0], y_test, color='black')


# In[26]:


plt.scatter(X_test[:,0], y_test, color='black')
plt.scatter(X_test[:,0],clf.predict(X_test))


# In[27]:


correlation_matrix = df.corr().nlargest(6,'SalePrice')


# In[28]:


correlation_matrix.axes[0]


# In[29]:


import seaborn as sns
ax = sns.heatmap(correlation_matrix, annot=True)


# In[30]:


train_data_cor = pd.DataFrame(df[correlation_matrix.axes[0]].drop(['SalePrice'],axis='columns'))
train_label = pd.DataFrame(df['SalePrice'])


# In[31]:


train_data_cor


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(train_data_cor,train_label,test_size=0.3)
clf = Ridge(alpha=1)
clf.fit(X_train,y_train)


# In[33]:


clf.score(X_test,y_test)


# In[34]:


mean_squared_error(y_test,clf.predict(X_test))


# In[35]:


linearR_model = LinearRegression()
linearR_model.fit(X_train,y_train)
linear_result = linearR_model.predict(X_test)

print("linear regression model score:")
linearR_model.score(X_test,y_test)


# In[36]:


mean_squared_error(y_test,linear_result)


# In[37]:


plt.scatter(X_test.iloc[:,0], y_test.iloc[:,0], color='black')


# In[38]:


plt.scatter(X_test.iloc[:,0], y_test.iloc[:,0], color='black')
plt.scatter(X_test.iloc[:,0],clf.predict(X_test))


# In[39]:


correlation_matrix = df.corr().nsmallest(5,'SalePrice')


# In[40]:


correlation_matrix


# In[41]:


train_data_cor = pd.DataFrame(df[correlation_matrix.axes[0]])


# In[47]:


train_data_cor


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(train_data_cor,train_label,test_size=0.3)
clf = Ridge(alpha=1)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


# In[49]:


mean_squared_error(y_test,clf.predict(X_test))


# In[50]:


plt.scatter(X_test.iloc[:,0], y_test.iloc[:,0], color='black')
plt.scatter(X_test.iloc[:,0],clf.predict(X_test))


# In[51]:


plt.scatter(train_data['GrLivArea'],df['SalePrice'])


# In[52]:


#find outliers
Q1=df.quantile(0.4)
Q3=df.quantile(0.6)
IQR=Q3-Q1
o=(df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))
o['GrLivArea']
#True == outliers


# In[53]:


#now get rid of outliers
nooutlier = df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR)))]


# In[54]:


plt.scatter(nooutlier['GrLivArea'],nooutlier['SalePrice'])


# In[ ]:




