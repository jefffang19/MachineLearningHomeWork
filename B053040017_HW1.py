#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import matplotlib.pyplot as plt


# In[2]:


def makecords(co=[]):
    #the equation looks like co[0]x^3 + co[1]x^2 + co[2]x^1 +co[3]
    #so let x = 1~10, gap = 0.1, Get each y's values
    y = []
    for i in np.arange(1,10,0.1):
        y.append(co[0]*(i**3) + co[1]*(i**2) + co[2]*(i**1) + co[3])
    
    #return each y's values
    return y    


# In[18]:


#differential of co[0]x^3 + co[1]x^2 + co[2]x^1 +co[3]
def diff(co, x):
    return 3*co[0]*(x**2)+2*co[1]*x+co[2]


# In[4]:


isEnd=False
while(not isEnd):
    try:
        #if not every element is int, goto except
        co=eval(input("Input the coefficient of third-order equation: (format a, b, c, d)"))
        #check if num of coefficient is correct
        if(not isinstance(co,int) and len(co)==4):
            isEnd=True
        else: isEnd=False
        if(isEnd): break
        print("Input Format Error")
    except:
        print("Input Format Error")

    
line=plt.plot(np.arange(1,10,0.1),makecords(co))
plt.setp(line,color='purple')
plt.show()


# In[7]:


while(1):
    try:
        nodes=eval(input("Input the coordinates of five points:"))
        #check if num of coefficient is correct
        if(not isinstance(nodes,int) and len(nodes)==10):
            isEnd=True
        else: isEnd=False
        if(isEnd): break
        print("Input Format Error")
    except:
        print("Input Format Error")

line=plt.plot(np.arange(1,10,0.1),makecords(co),label='equation')
plt.setp(line,color='purple')
for i in range(0,9,2):
    #give the dots corresponding color according to the y's value
    if(nodes[i+1] > co[0]*(nodes[i]**3) + co[1]*(nodes[i]**2) + co[2]*(nodes[i]**1) + co[3]):
        plt.plot(nodes[i],nodes[i+1],'go')
    elif(nodes[i+1] < co[0]*(nodes[i]**3) + co[1]*(nodes[i]**2) + co[2]*(nodes[i]**1) + co[3]):
        plt.plot(nodes[i],nodes[i+1],'bo')
    elif(nodes[i+1] == co[0]*(nodes[i]**3) + co[1]*(nodes[i]**2) + co[2]*(nodes[i]**1) + co[3]):
        plt.plot(nodes[i],nodes[i+1],'ro')
plt.legend(loc='upper right')
plt.title("My Graph")
plt.show()


# In[27]:


#find maximan in range x=0~10
x=1
learning_rate=0.01
epn=0.00001
while(1):
    x_new=x+learning_rate*(diff(co,x))
    if(x_new-x)<epn:
        break
    elif(x_new>=10): break 
    #print(x_new)
    x=x_new
    


print("y max happens when x= ",x)


# In[ ]:




