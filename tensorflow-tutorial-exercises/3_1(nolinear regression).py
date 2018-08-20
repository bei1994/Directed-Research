import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x_data = np.linspace(-.5, .5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = x_data**2 + noise


# define placeholder
x = tf.placeholder(tf.float32, shape=[None, 1], name='input')
y = tf.placeholder(tf.float32, shape=[None, 1], name='label')

# define one hidden layer 10 neurons
weights_L1 = tf.Variable(tf.random_normal(shape=[1, 10]))
bias_L1 = tf.Variable(tf.zeros([1, 10]))
wx_plus_b_L1 = tf.matmul(x, weights_L1) + bias_L1
L1 = tf.nn.tanh(wx_plus_b_L1)

# define output layer 1 output
weights_L2 = tf.Variable(tf.random_normal(shape=[10, 1]))
bias_L2 = tf.Variable(tf.zeros([1, 1]))
wx_plus_b_L2 = tf.matmul(L1, weights_L2) + bias_L2
y_predic = tf.nn.tanh(wx_plus_b_L2)

# define loss function and optimizer
loss = tf.reduce_mean(tf.square(y - y_predic))
optimizer = tf.train.GradientDescentOptimizer(.1)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2001):
        sess.run(train, feed_dict={x: x_data, y: y_data})

    # analyse model
    predic_value = sess.run(y_predic, feed_dict={x: x_data})
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, predic_value, 'r-', lw=5)
    plt.show()
