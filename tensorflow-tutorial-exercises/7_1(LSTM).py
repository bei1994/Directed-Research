import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# retrieve dataset
mnist = input_data.read_data_sets("MNIST", one_hot=True)

# define basic paras
num_input = 28
max_time = 28
num_uints = 100
num_class = 10
batch_size = 50
nbatch = mnist.train.num_examples // batch_size

# define placeholder to feed data
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, num_class])

# initialize weights and biases
weights = tf.Variable(tf.truncated_normal([num_uints, num_class], stddev=.1))
biases = tf.Variable(tf.zeros([num_class]) + .1)


# define LSTM network
def LSTM(X, weights, biases):
    # input shape: [batch_size, max_time, num_input]
    inputs = tf.reshape(X, [-1, max_time, num_input])
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_uints)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    # outputs = tf.unstack(outputs, axis=1)
    result = tf.matmul(final_state[1], weights) + biases
    return result


# calculate loss
logits = LSTM(x, weights, biases)

# define cross_entropy loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# calculate test accuracy
predic_result = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(predic_result, tf.float32))


saver = tf.train.Saver()

# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("logs/", sess.graph)
    for epoch in range(21):
        for batch in range(nbatch):
            batch_samples, batch_labels = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_samples,
                                            y: batch_labels})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                            y: mnist.test.labels})

        print("epoch: %2d, test accuracy: %6.4f" % (epoch, acc))

    # save trained models and sess
    saver.save(sess, "net/LSTM/lstm_net.ckpt")
