import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# retrieve dataset
mnist = input_data.read_data_sets("MNIST", one_hot=True)

# define batch size
batch_size = 100
nbatch = mnist.train.num_examples // batch_size

# defien weight initialization


def weight_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=.1, name=name))

# define bias initialization


def bias_variable(shape, name):
    return tf.Variable(tf.constant(.1, shape=shape, name=name))


# define conv layers
# samples: [batch, image_height, image_width, in_channel]
# filters: [filter_height, filter_width, in_channel, out_channel]
# strides: [1, stride in d1, stride in d2, 1]
# padding: "SAME" for padding 0 around border; "VALID" for no padding
def conv2d(samples, filters):
    return tf.nn.conv2d(input=samples, filter=filters,
                        strides=[1, 1, 1, 1], padding='SAME')

# define max pooling layer
# x: 4D default is [batch, image_height, image_width, in_channel]("NHWC")
# ksize: [1, filter_height, filter_width, 1]


def max_pooling(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


with tf.name_scope("input"):
    # define placeholder to feed data
    samples = tf.placeholder(tf.float32, [None, 784])
    labels = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    with tf.name_scope("image"):
        # reshape input samples into 4D[batch, height, width, channel]
        input_data = tf.reshape(samples, [-1, 28, 28, 1])

with tf.name_scope("Conv1"):
    # first conv layer and max pooling layer
    # filter size: 5*5*1, number: 32
    with tf.name_scope("W_Conv1"):
        W_conv1 = weight_variable([5, 5, 1, 32], name="W_Conv1")
    with tf.name_scope("b_Conv1"):
        b_conv1 = bias_variable([32], name="b_conv1")
    with tf.name_scope("conv2d_1"):
        conv2d_1 = conv2d(input_data, W_conv1) + b_conv1
    with tf.name_scope("relu"):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope("pooling_1"):
        h_pool1 = max_pooling(h_conv1)

with tf.name_scope("Conv2"):
    # second conv layer and max pooling layer
    # filter size: 5*5*32, number: 64
    with tf.name_scope("W_Conv2"):
        W_conv2 = weight_variable([5, 5, 32, 64], name="W_conv2")
    with tf.name_scope("b_Conv2"):
        b_conv2 = bias_variable([64], name="b_conv2")
    with tf.name_scope("conv2d_2"):
        conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    with tf.name_scope("relu"):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope("pooling_2"):
        h_pool2 = max_pooling(h_conv2)

# h_pool2 size: [batch, 7, 7, 64]

with tf.name_scope("fc1"):
    # first fully connected layer, 1024 neurons
    # weights: [inputs, neurons]
    # bias: [neurons]
    with tf.name_scope("W_fc1"):
        W_fc1 = weight_variable([7 * 7 * 64, 1024], name="W_fc1")
    with tf.name_scope("b_fc1"):
        b_fc1 = bias_variable([1024], name="b_fc1")
    with tf.name_scope("h_pool2_flat"):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name="h_pool2_flat")
    with tf.name_scope("wx_plus_b1"):
        wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    with tf.name_scope("relu"):
        h_fc1 = tf.nn.relu(wx_plus_b1)
    with tf.name_scope("fc1_drop"):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope("fc2"):
    # second fully connected layer, 10 neurons
    # weights: [inputs, neurons]
    # bias: [neurons]
    with tf.name_scope("W_fc2"):
        W_fc2 = weight_variable([1024, 10], name="W_fc2")
    with tf.name_scope("b_fc2"):
        b_fc2 = bias_variable([10], name="b_fc2")
    with tf.name_scope("wx_plus_b2"):
        wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    with tf.name_scope("softmax"):
        logits = tf.nn.softmax(wx_plus_b2)

with tf.name_scope("loss"):
    # define cross_entropy loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    tf.summary.scalar("loss", loss)

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
# train_step = tf.train.AdamOptimizer(.001).minimize(loss)

with tf.name_scope("accuracy"):
    with tf.name_scope("predic_result"):
        # calculate test accuracy
        predic_result = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(predic_result, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

merged = tf.summary.merge_all()

# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter("logs/train", sess.graph)
    test_writer = tf.summary.FileWriter("logs/test", sess.graph)

    for iter in range(1001):
        batch_samples, batch_labels = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={samples: batch_samples, labels: batch_labels, keep_prob: .5})
        summary = sess.run(merged, feed_dict={samples: batch_samples, labels: batch_labels, keep_prob: 1.0})
        train_writer.add_summary(summary, iter)

        batch_samples, batch_labels = mnist.test.next_batch(batch_size)
        summary = sess.run(merged, feed_dict={samples: batch_samples, labels: batch_labels, keep_prob: 1.0})
        test_writer.add_summary(summary, iter)

        if iter % 100 == 0:
            test_acc = sess.run(accuracy, feed_dict={samples: mnist.test.images,
                                                     labels: mnist.test.labels,
                                                     keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={samples: mnist.train.images,
                                                      labels: mnist.train.labels,
                                                      keep_prob: 1.0})

            print("Iter: %4d, test accuracy: %6.4f, train accuracy: %6.4f" %
                  (iter, test_acc, train_acc))
