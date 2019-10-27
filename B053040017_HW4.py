#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[2]:


import numpy as np


# In[3]:


import random


# In[4]:


#prepare data
#f(x)= x**3 + x**2 - x - 1
X = tf.placeholder(tf.float32)
Y = X**3+X**2-X-1

train_x = list(np.arange(-100,100,0.1))
random.shuffle(train_x)

with tf.Session() as sess:
    train_y = sess.run(Y, {X:train_x})
    #print(train_x,train_y)


# In[5]:


#some training args
nsteps = 500
batch_size = 4
show_step = True
lr = 0.1

#num of units in hidden layer
h1 = 200
h2 = 400


# In[6]:


def fc_layer(x, n_units, name):
    input_dim = x.get_shape()[1]
    #print out hidden layer shape
    print('input shape '+name+'= '+str(input_dim))
    w = tf.get_variable("W"+name, dtype=tf.float32, shape=[input_dim, n_units], initializer=tf.truncated_normal_initializer(stddev=0.01))
    b = tf.get_variable('b'+name, dtype=tf.float32, initializer=tf.constant(0., shape=[n_units], dtype=tf.float32))
    return tf.matmul(x,w)+b

    


# In[7]:


def nnet(x):
    layer1 = fc_layer(x, h1, 'fc1')
    layer1 = tf.nn.relu(layer1)
    layer2 = fc_layer(layer1, h2, 'fc2')
    layer2 = tf.nn.relu(layer2)
    layerout = fc_layer(layer2, 1, 'fco')
    return layerout


# In[8]:


#input tensor
X = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='X')
y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')

#construct model
logits = nnet(X)


# In[9]:


loss_opt = tf.reduce_mean(tf.square(y-logits))
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
train = optimizer.minimize(loss_opt)

correct_predict = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))


# In[10]:


def cut_batch(x,y,size,index):
    sub_x = []
    sub_y = []
    for i in range(size):
        cur = size*index+i
        if(cur < len(x)):
            sub_x.append(x[size*index+i])
            sub_y.append(y[size*index+i])
        else: break
        
    return sub_x, sub_y


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


fig,ax = plt.subplots()
fig.set_tight_layout(True)
ax.scatter(train_x,train_y)


# In[19]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(nsteps):
        batch_x, batch_y = cut_batch(train_x, train_y, batch_size, i)
        batch_x = np.array(batch_x,float)[:,np.newaxis]
        for i in range(len(batch_y)):
            #add noise
            batch_y[i]+=np.random.normal(0,0.09)
        batch_y = np.array(batch_y,float)[:,np.newaxis]
        
        
        #debug
        print(batch_x,batch_y)
        
        sess.run(train,{X:batch_x, y:batch_y})
        if(show_step):
            loss, acc = sess.run([loss_opt, accuracy], {X:batch_x, y:batch_y})
            print("Step " + str(i+1) + " , Loss= " + str(loss) + " , Acc= " + str(acc))
        
    print("Train Finished")
    
    trainx = np.array(train_x,float)[:,np.newaxis]
    trainy = np.array(train_y,float)[:,np.newaxis]
    
    print("Testing Acc:", sess.run(accuracy, {X: trainx, y: trainy}))
    
    #plot result
    plt.scatter(trainx,trainy,color='blue')
    plt.scatter(trainx,sess.run(logits,{X:trainx}), color='green')
    plt.show()
    


# In[18]:


x = [1,2,3,4]
for i in range(len(x)):
    x[i] +=1
print(x)


# In[15]:


#keras
def build_model():
        model = keras.Sequential()
        model.add(keras.layers.Dense(100, input_shape=(1,),activation='sigmoid'))
        model.summary()
        return model

    
model = build_model()
#sgd=keras.optimizers.SGD(lr=0.1,decay=1e-2, momentum=0.9)
model.compile(loss='mean_squared_error',optimizer="sgd")
model.fit([x], [y], epochs=100, verbose=1)
score = model.evaluate([x],[y])

plt.scatter(x, y,color='green')
plt.plot(x, model.predict([x]))


# In[ ]:


plt.scatter(x, y,color='green')
plt.scatter(x, model.predict([x]))

