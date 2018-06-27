

import tensorflow as tf
import numpy as np


data = np.loadtxt('./data.csv',delimiter=',',unpack=True,dtype='float32')


x_data = np.transpose(data[0:3])
y_data = np.transpose(data[3:])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


W1 = tf.Variable(tf.random_uniform([3,50],-1.,1.))
W2 = tf.Variable(tf.random_uniform([50,100],-1.,1.))
W3 = tf.Variable(tf.random_uniform([100,3],-1.,1.))


L1 = tf.sigmoid(tf.matmul(X,W1))
L2 = tf.sigmoid(tf.matmul(L1,W2))
L3 = tf.matmul(L2,W3)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=L3,labels= Y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)


with tf.Session() as sess:
  
    init = tf.global_variables_initializer()
    sess.run(init)
   
    for step in range(20001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        
        if step % 1000 == 0:
            print (step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
        
   
    pred = tf.argmax(L3,1)
    real = tf.argmax(Y,1)

    print("Prediction:",sess.run(pred,feed_dict={X:x_data}))
    print("Real:",sess.run(real, feed_dict={Y:y_data}))

   
    print("Grade: ",sess.run(pred,feed_dict={X:[[80,80,80]]}))








