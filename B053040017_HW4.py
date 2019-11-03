#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
print("Using TensorFlow Version %s" %tf.__version__)


# In[3]:


import numpy as np


# In[4]:


import random


# In[17]:


#prepare data
#f(x)= x**3 + x**2 - x - 1
X = tf.placeholder(tf.float32)
Y = X**3+X**2-X-1

train_x = list(np.arange(-100,100,0.1))
random.shuffle(train_x)

with tf.Session() as sess:
    #add noise
    train_y = sess.run(Y, {X:train_x})+np.random.normal(0,0.09)
    #print(train_x,train_y)


# In[18]:


#some training args
nsteps = 300
#batch_size = 4
show_step = True
lr = 0.1

#num of units in hidden layer
h1 = 100
h2 = 200


# In[19]:


def fc_layer(x, n_units, name):
    input_dim = x.get_shape()[1]
    #print out hidden layer shape
    print('input shape '+name+'= '+str(input_dim))
    w = tf.get_variable("W"+name, dtype=tf.float32, shape=[input_dim, n_units], initializer=tf.truncated_normal_initializer(stddev=0.01))
    b = tf.get_variable('b'+name, dtype=tf.float32, initializer=tf.constant(0., shape=[n_units], dtype=tf.float32))
    return tf.matmul(x,w)+b

    


# In[20]:


def nnet(x,name):
    layer1 = fc_layer(x, h1, 'fc1'+str(name))
    layer1 = tf.nn.relu(layer1)
    layer2 = fc_layer(layer1, h2, 'fc2'+str(name))
    layer2 = tf.nn.relu(layer2)
    layerout = fc_layer(layer2, 1, 'fco'+str(name))
    return layerout


# In[21]:


#input tensor
X = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='X')
y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')

#construct model
logits = nnet(X,'a')


# In[22]:


loss_opt = tf.reduce_mean(tf.square(y-logits))
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
train = optimizer.minimize(loss_opt)

correct_predict = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))


# In[23]:


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


# In[24]:


import matplotlib.pyplot as plt


# In[25]:


fig,ax = plt.subplots()
fig.set_tight_layout(True)
ax.scatter(train_x,train_y)


# In[26]:


#train data input format config
train_x = np.array(train_x,float)[:,np.newaxis]
train_y = np.array(train_y,float)[:,np.newaxis]


# In[27]:


predict_a = []

    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(nsteps):
        '''
        batch_x, batch_y = cut_batch(train_x, train_y, batch_size, i)
        batch_x = np.array(batch_x,float)[:,np.newaxis]
        for j in range(len(batch_y)):
            #add noise
            batch_y[j]+=np.random.normal(0,0.09)
        batch_y = np.array(batch_y,float)[:,np.newaxis]
        '''
        
        
        _,loss,acc = sess.run([train,loss_opt,accuracy],{X:train_x, y:train_y})
        
        if(show_step):
            print("Step " + str(i+1) + " , Loss= " + str(loss))
        
    print("Train Finished")
    
    predict_a = sess.run(logits,{X:train_x})
    


# In[28]:


#plot result
print("hidden layer 1: "+str(h1)+"\nhidden layer 2: "+str(h2))
plt.scatter(train_x,train_y,color='blue')
plt.scatter(train_x,predict_a, color='green')
plt.show()


# In[29]:


#num of units in hidden layer
h1 = 45
h2 = 100

#construct model
logits = nnet(X,'b')


# In[30]:


loss_opt = tf.reduce_mean(tf.square(y-logits))
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
train = optimizer.minimize(loss_opt)

correct_predict = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))


# In[31]:


predict_b = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(nsteps):
        '''
        batch_x, batch_y = cut_batch(train_x, train_y, batch_size, i)
        batch_x = np.array(batch_x,float)[:,np.newaxis]
        for j in range(len(batch_y)):
            #add noise
            batch_y[j]+=np.random.normal(0,0.09)
        batch_y = np.array(batch_y,float)[:,np.newaxis]
        '''
        
        
        _,loss,acc = sess.run([train,loss_opt,accuracy],{X:train_x, y:train_y})
        
        if(show_step):
            print("Step " + str(i+1) + " , Loss= " + str(loss))
        
    print("Train Finished")
    
    predict_b = sess.run(logits,{X:train_x})
    


# In[32]:


#plot result
print("hidden layer 1: "+str(h1)+"\nhidden layer 2: "+str(h2))
plt.scatter(train_x,train_y,color='blue')
plt.scatter(train_x,predict_b, color='green')
plt.show()


# In[33]:


#num of units in hidden layer
h1 = 10
h2 = 30

#construct model
logits = nnet(X,'c')


# In[34]:


loss_opt = tf.reduce_mean(tf.square(y-logits))
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
train = optimizer.minimize(loss_opt)

correct_predict = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))


# In[35]:


predict_c = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(nsteps):
        '''
        batch_x, batch_y = cut_batch(train_x, train_y, batch_size, i)
        batch_x = np.array(batch_x,float)[:,np.newaxis]
        for j in range(len(batch_y)):
            #add noise
            batch_y[j]+=np.random.normal(0,0.09)
        batch_y = np.array(batch_y,float)[:,np.newaxis]
        '''
        
        
        _,loss,acc = sess.run([train,loss_opt,accuracy],{X:train_x, y:train_y})
        
        if(show_step):
            print("Step " + str(i+1) + " , Loss= " + str(loss))
        
    print("Train Finished")
    
    predict_c = sess.run(logits,{X:train_x})
    


# In[36]:


#plot result
print("hidden layer 1: "+str(h1)+"\nhidden layer 2: "+str(h2))
plt.scatter(train_x,train_y,color='blue')
plt.scatter(train_x,predict_c, color='green')
plt.show()


# In[37]:


print("HW4 Part 1 comparsion")
print("hidden layer 1: 10\nhidden layer 2: 30\ngreen\n")
print("hidden layer 1: 45\nhidden layer 2: 100\nyellow\n")
print("hidden layer 1: 100\nhidden layer 2: 200\nred\n")
plt.scatter(train_x,train_y,color='blue')
plt.scatter(train_x,predict_c,color='green')
plt.scatter(train_x,predict_b,color='yellow')
plt.scatter(train_x,predict_a,color='red')


# In[38]:


from sklearn import datasets
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow.contrib.layers import fully_connected
get_ipython().run_line_magic('matplotlib', 'inline')
print("Using TensorFlow Version %s" %tf.__version__)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  


# In[39]:


np.random.seed(0)
X, Y = datasets.make_circles(n_samples=500, factor=0.1,
noise=0.1)

# Split into test and training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.25,
                                                    random_state=73)


# In[7]:


# Define network dimensions
n_inputs = X_train.shape[0]
n_input_dim = X_train.shape[1]
# Layer size
n_hidden = 4 # Number of hidden nodes
n_output = 1 # Number of output nodes = for binary classifier


# In[8]:


X_input = tf.placeholder(tf.float32, [None, n_input_dim], name='input')
y = tf.placeholder(tf.float32, [None, n_output], name='y')


# In[9]:


initializer = tf.contrib.layers.xavier_initializer()
hidden1 = fully_connected(X_input, n_hidden, activation_fn=tf.nn.elu,
                         weights_initializer=initializer)
logits = fully_connected(hidden1, n_output, activation_fn=tf.nn.sigmoid,
                        weights_initializer=initializer)


# In[13]:


loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
cost = tf.reduce_mean(loss)
learning_rate = 0.01
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(cost)


# In[16]:


# Reshape labels to match placeholder
Y_train = Y_train.reshape(-1, 1)

# Define feed dicts
train_feed = {X_input: X_train,
              y: Y_train}
test_feed = {X_input: X_test}

# Initialize list to store cost results
iter_cost = []
iters = 1000

# Initialize global variables
init = tf.global_variables_initializer()
# Start session and run loop
with tf.Session() as sess:
    sess.run(init)
    
    # Run training loop
    for i in range(iters):
        _, cost_ = sess.run([train_op, cost],
                       feed_dict=train_feed)
        
        # Append the cost
        iter_cost.append(cost_)
        
    # Plot training loss
    plt.figure(figsize=(10,8))
    plt.plot(iter_cost)
    plt.title("Training Loss")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Mean Cost")
    plt.show()   

    # Make prediction
    y_prob = sess.run([logits],
                    feed_dict=test_feed)[0]
    
    # Replace probabilities with lables for comparison
    y_hat = np.where(y_prob<0.5,0,1)
    # Get prediction accuracy
    acc = np.sum(Y_test.reshape(-1,1)==y_hat) / len(Y_test)
    print("Test Accuracy %.2f" %acc)
    
    # Plot the decision boundary
    # Determine grid range in x and y directions
    x_min, x_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
    y_min, y_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1
    
    # Set grid spacing parameter
    spacing = min(x_max - x_min, y_max - y_min) / 100

    # Create grid
    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                   np.arange(y_min, y_max, spacing))
    
    # Concatenate data to match input
    data = np.hstack((XX.ravel().reshape(-1,1), 
                      YY.ravel().reshape(-1,1)))

    # Pass data to predict method
    db_prob = sess.run([logits],
                      feed_dict={X_input: data})[0]
    clf = np.where(db_prob<0.5,0,1)
    
    Z = clf.reshape(XX.shape)

    plt.figure(figsize=(10,8))
    plt.contourf(XX, YY, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c=Y, 
                cmap=plt.cm.Spectral)
    plt.show()


# In[ ]:




