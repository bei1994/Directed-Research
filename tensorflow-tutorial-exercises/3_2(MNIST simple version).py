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

# define network
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros(10))
logits = tf.nn.softmax(tf.matmul(samples, W) + b)

# define quadratic loss and train
# loss = tf.reduce_mean(tf.square(logits - labels))

# define cross_entropy loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
train_step = tf.train.GradientDescentOptimizer(.2).minimize(loss)
# train_step = tf.train.AdamOptimizer(.001).minimize(loss)

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
                                            labels: batch_labels})

        acc = sess.run(accuracy, feed_dict={samples: mnist.test.images,
                                            labels: mnist.test.labels})

        print("epoch: %2d, test accuracy: %6.4f" % (epoch, acc))
