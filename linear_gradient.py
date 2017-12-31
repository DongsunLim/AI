import tensorflow as tf
x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#Out hypothesis for Linear model X*W
hypothesis = X*W

# cost/Loss function
cost = tf.reduce_sum(tf.square(hypothesis - Y))

learning_rate = 0.1
gradient = tf.reduce_mean((W*X - Y)*X)
descent = W - learning_rate * gradient
update = W.assign(descent)

sess = tf.Session()
#Change global_variables_initializer to initialize_all_variables from sample because of error
sess.run(tf.initialize_all_variables())
for step in range(21):
	sess.run(update, feed_dict={X: x_data, Y: y_data})
	print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

