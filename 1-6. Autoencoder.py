
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./tmp/data/", one_hot=True)


learning_rate = 0.01
training_epoch = 15
batch_size = 100


n_hidden = 300


n_input = 28*28

# Step 1 Neural network setting

X = tf.placeholder(tf.float32, [None, n_input])

# input -> encoder -> decoder -> output

W1 = tf.Variable(tf.random_normal([n_input,n_hidden]))
B1 = tf.Variable(tf.random_normal([n_hidden]))

W2 = tf.Variable(tf.random_normal([n_hidden,n_input]))
B2 = tf.Variable(tf.random_normal([n_input]))

encoder = tf.nn.sigmoid(tf.add(tf.matmul(X,W1),B1))
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder,W2),B2))


Y = X

cost = tf.reduce_mean(tf.pow(Y - decoder,2))
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

total_batch = int(mnist.train.num_examples/batch_size)


with tf.Session() as sess:
    init = tf.global_variables_initializer()

    sess.run(init)

    for epoch in range(training_epoch):
        sum_cost = 0

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={X:batch_xs})
            sum_cost += sess.run(cost,feed_dict={X:batch_xs})

        print("Epoch:",epoch,"Avg Cost:",sum_cost/total_batch)
    
    print("Optimization finished")

    # Decoding

    pred = sess.run(decoder,feed_dict={X:mnist.test.images[:10]})
    figure, axis = plt.subplots(2,10,figsize=(10,2))

    for i in range(10):
        axis[0][i].set_axis_off()
        axis[1][i].set_axis_off()
        axis[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        axis[1][i].imshow(np.reshape(pred[i],(28,28)))

    plt.show()


