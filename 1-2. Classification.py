
import tensorflow as tf
import numpy as np


x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([[0],[1],[1],[0]])



X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


W1 = tf.Variable(tf.random_uniform([2,30],-1.0,1.0))
W2 = tf.Variable(tf.random_uniform([30,1],-1.0,1.0))

 
b1 = tf.Variable(tf.zeros([30]))
b2 = tf.Variable(tf.zeros([1]))


L1 = tf.sigmoid(tf.add(tf.matmul(X,W1),b1))
L2 = tf.sigmoid(tf.add(tf.matmul(L1,W2),b2))
hypothesis = L2



cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


with tf.Session() as sess:
 
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(100001):
        sess.run(train,feed_dict={X:x_data,Y:y_data})
       
        if step % 10000 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data,Y:y_data}))


    pred = tf.floor(hypothesis+0.5)
    real = Y
 
    print("Prediction:",sess.run(pred,feed_dict={X:x_data}))
    print("Real:",sess.run(real,feed_dict={Y:y_data}))

