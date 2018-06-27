
import tensorflow as tf



from tensorflow.examples.tutorials.mnist import input_data


# mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
mnist = input_data.read_data_sets("./tmp/data/", one_hot=True)



X = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

# (1 x 784) x (784 x 300) x (300 x 512) x (512 x 1024) x (1024 x 10) = (1 x 10) 

W1 = tf.Variable(tf.random_normal([784,300],stddev = 0.01))
W2 = tf.Variable(tf.random_normal([300,512],stddev = 0.01))
W3 = tf.Variable(tf.random_normal([512,1024],stddev = 0.01))
W4 = tf.Variable(tf.random_normal([1024,10],stddev = 0.01))

L1 = tf.nn.sigmoid(tf.matmul(X,W1))
L2 = tf.nn.sigmoid(tf.matmul(L1,W2))
L3 = tf.nn.sigmoid(tf.matmul(L2,W3))
hypothesis = tf.matmul(L3,W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= hypothesis,labels= Y))

train = tf.train.AdamOptimizer(0.001).minimize(cost)


with tf.Session() as sess:
  
    init = tf.global_variables_initializer()
    sess.run(init)
    batch_size = 100
    total_batch = int(mnist.train.num_examples/batch_size)

  
    for step in range(20):
    
        sum_cost = 0
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={X: batch_xs, Y: batch_ys})
            sum_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys})

        print("Step:",step,"Average cost:",sum_cost/total_batch)

    print("Optimization Finished")

    pred = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(pred,tf.float32))
  
    print("Accuracy:",sess.run(accuracy,feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
