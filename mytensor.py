import tensorflow as tf

#X and Y data
x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable (tf.random_normal([1]), name = 'weight')
b = tf.Variable (tf.random_normal([1]), name = 'bias')

# Out hypothesis XW+b
hypothesis = x_train*W +b

# cost/Loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

#Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Fit the line
for step in range(2001):
	sess.run(train)
	if step%20==0:
		print(step, sess.run(cost), sess.run(W), sess.run(b))


