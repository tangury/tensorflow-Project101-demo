import tensorflow as tf



x_data = [1, 2, 3, 4, 5, 6, 7]
y_data = [1, 2, 3, 4, 5, 6, 7]

W = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = tf.add(tf.multiply(X,W),b)

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

train = optimizer.minimize(cost)


with tf.Session() as sess:
   
    init = tf.global_variables_initializer()
    sess.run(init)
  
    for step in range(10000):
       
        sess.run(train, feed_dict={X: x_data, Y: y_data})
      
        if step % 100 == 0:
            print(step, sess.run(cost,feed_dict={X:x_data,Y:y_data}), sess.run(W), sess.run(b))


  
    print("Training Finished")
    print("X: 10, Y:", sess.run(hypothesis, feed_dict={X: 10}))
    print("X: 13, Y:", sess.run(hypothesis, feed_dict={X: 13}))
