#!/usr/bin/env python
# coding: utf-8

# In[2]:


#tensorflow version 1.13.1
import tensorflow as tf


# In[3]:


from tensorflow import keras


# In[4]:


import numpy as np


# In[5]:


import random


# In[1]:


#prepare data
#f(x)= x**3 + x**2 - x - 1
X = tf.placeholder(tf.float32)
Y = X**3+X**2-X-1

train_x = list(np.arange(-10,10,0.1))
random.shuffle(train_x)

with tf.Session() as sess:
    train_y = sess.run(Y, {X:train_x})
    print(train_x,train_y)


# In[7]:


import matplotlib.pyplot as plt


# In[ ]:


#keras
def build_model():
        model = keras.Sequential()
        model.add(keras.layers.Dense(100, input_shape=(1,),activation='sigmoid'))
        model.summary()
        return model

    
model = build_model()
#sgd=keras.optimizers.SGD(lr=0.1,decay=1e-2, momentum=0.9)
model.compile(loss='mean_squared_error',optimizer="sgd")
model.fit(train_x, train_y, epochs=100, verbose=1)
score = model.evaluate(train_x,train_y)

plt.scatter(train_x, train_y,color='green')
plt.plot(train_x, model.predict(train_x))


# In[ ]:


plt.scatter(train_x, train_y,color='green')
plt.scatter(train_x, model.predict(train_x))


# In[ ]:


#now build NN


layer = [1,128,256,1]

weights = {
    'w1': tf.Variable(tf.random_normal([layer[0],layer[1]])),
    'w2': tf.Variable(tf.random_normal([layer[1],layer[2]])),
    'out': tf.Variable(tf.random_normal([layer[2],layer[3]]))
}
bias = {
    'b1': tf.Variable(tf.random_normal([layer[1]])),
    'b2': tf.Variable(tf.random_normal([layer[2]])),
    'out': tf.Variable(tf.random_normal([layer[3]]))
}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(weights['w1'])
    print(sess.run(bias['out']))
    #print(sess.run(weights['w1']))


# In[ ]:


def nnet(x):
    layer1 = tf.add(tf.matmul)

