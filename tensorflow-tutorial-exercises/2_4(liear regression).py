import tensorflow as tf
import numpy as np 

np.random.seed(0)
x_data = np.random.rand(100)
y_data = 0.1 * x_data + 0.2

k = tf.Variable(.0)
b = tf.Variable(.0)
y_predi = k * x_data + b

# define loss function
loss = tf.reduce_mean(tf.square(y_predi - y_data))
# define optimizer
optimizer = tf.train.GradientDescentOptimizer(.72)
# training
train = optimizer.minimize(loss)

# initialize variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for step in range(201):
		sess.run(train)
		if step%20 == 0:
			print("step: %3d, k: %10.8f, b: %10.8f, loss: %14.12f"  % 
				        (step, sess.run(k), sess.run(b), sess.run(loss)))

