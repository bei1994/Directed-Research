import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# retrieve dataset
mnist = input_data.read_data_sets("MNIST", one_hot=True)

# define batch size
batch_size = 100
nbatch = mnist.train.num_examples // batch_size

# define placeholder to feed data
samples = tf.placeholder(tf.float32, [None, 784])
labels = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# define network
W1 = tf.Variable(tf.truncated_normal(shape=[784, 2000], stddev=.1))
b1 = tf.Variable(tf.zeros(2000) + .1)
L1 = tf.nn.tanh(tf.matmul(samples, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal(shape=[2000, 2000], stddev=.1))
b2 = tf.Variable(tf.zeros(2000) + .1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal(shape=[2000, 1000], stddev=.1))
b3 = tf.Variable(tf.zeros(1000) + .1)
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
L3_drop = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.truncated_normal(shape=[1000, 10], stddev=.1))
b4 = tf.Variable(tf.zeros(10) + .1)
logits = tf.nn.softmax(tf.matmul(L3_drop, W4) + b4)

# define quadratic loss and train
# loss = tf.reduce_mean(tf.square(logits - labels))

# define cross_entropy loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
train_step = tf.train.GradientDescentOptimizer(.2).minimize(loss)

# calculate test accuracy
logits_result = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(logits_result, tf.float32))

# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(nbatch):
            batch_samples, batch_labels = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={samples: batch_samples,
                                            labels: batch_labels,
                                            keep_prob: .7})

        test_acc = sess.run(accuracy, feed_dict={samples: mnist.test.images,
                                                 labels: mnist.test.labels,
                                                 keep_prob: 1.0})

        train_acc = sess.run(accuracy, feed_dict={samples: mnist.train.images,
                                                  labels: mnist.train.labels,
                                                  keep_prob: 1.0})

        print("epoch: %2d, test accuracy: %6.4f,train accuracy: %6.4f" %
              (epoch, test_acc, train_acc))
