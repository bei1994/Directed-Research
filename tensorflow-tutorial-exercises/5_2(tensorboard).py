import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# retrieve dataset
mnist = input_data.read_data_sets("MNIST", one_hot=True)

# define batch size
batch_size = 100
nbatch = mnist.train.num_examples // batch_size

# add name scope
with tf.name_scope(name="input"):
    # define placeholder to feed data
    samples = tf.placeholder(tf.float32, [None, 784], name="x_data")
    labels = tf.placeholder(tf.float32, [None, 10], name="y_data")

with tf.name_scope(name="layer"):
    # define network
    with tf.name_scope(name="weights"):
        W = tf.Variable(tf.zeros([784, 10]), name='W')
    with tf.name_scope(name="bias"):
        b = tf.Variable(tf.zeros(10), name='b')
    with tf.name_scope(name="wx_plus_b"):
        wx_plus_b = tf.matmul(samples, W) + b
    with tf.name_scope(name="softmax"):
        logits = tf.nn.softmax(wx_plus_b)

# define quadratic loss and train
# loss = tf.reduce_mean(tf.square(logits - labels))

with tf.name_scope("loss"):
    # define cross_entropy loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(.2).minimize(loss)
    # train_step = tf.train.AdamOptimizer(.001).minimize(loss)

with tf.name_scope("accuracy"):
    with tf.name_scope("logits"):
        # calculate test accuracy
        predic_result = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(predic_result, tf.float32))

# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("logs/", sess.graph)
    for epoch in range(1):
        for batch in range(nbatch):
            batch_samples, batch_labels = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={samples: batch_samples,
                                            labels: batch_labels})

        acc = sess.run(accuracy, feed_dict={samples: mnist.test.images,
                                            labels: mnist.test.labels})

        print("epoch: %2d, test accuracy: %6.4f" % (epoch, acc))
