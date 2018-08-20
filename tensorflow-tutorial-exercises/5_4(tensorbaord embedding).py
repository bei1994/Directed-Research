import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

# retrieve dataset
mnist = input_data.read_data_sets("MNIST", one_hot=True)

# define max steps
max_steps = 1001
# image nums
image_num = 3000
# file path
DIR = '/Users/liubei/Desktop/tensorflow exercise/'

# define session
sess = tf.Session()

# define embedding variable
embedding_var = tf.Variable(mnist.test.images[:image_num],
                            trainable=False, name="embedding")

# define para anallabelssis


def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)


# add name scope
with tf.name_scope(name="input"):
    # define placeholder to feed data
    samples = tf.placeholder(tf.float32, [None, 784], name="x_data")
    y = tf.placeholder(tf.float32, [None, 10], name="labels_data")

# add images to summaries
with tf.name_scope("images"):
    image_shaped_input = tf.reshape(samples, [-1, 28, 28, 1])
    tf.summary.image("input_image", image_shaped_input, 10)

with tf.name_scope(name="layer"):
    # define network
    with tf.name_scope(name="weights"):
        W = tf.Variable(tf.zeros([784, 10]), name='W')
        variable_summaries(W)
    with tf.name_scope(name="bias"):
        b = tf.Variable(tf.zeros(10), name='b')
        variable_summaries(b)
    with tf.name_scope(name="wx_plus_b"):
        wx_plus_b = tf.matmul(samples, W) + b
    with tf.name_scope(name="softmax"):
        logits = tf.nn.softmax(wx_plus_b)

# define quadratic loss and train
# loss = tf.reduce_mean(tf.square(logits - labels))

with tf.name_scope("loss"):
    # define cross_entroplabels loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    tf.summary.scalar("loss", loss)

with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(.5).minimize(loss)
    # train_step = tf.train.AdamOptimizer(.001).minimize(loss)

with tf.name_scope("accuracy"):
    with tf.name_scope("logits"):
        # calculate test accuraclabels
        predic_result = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    with tf.name_scope("accuracy"):
        accuraclabels = tf.reduce_mean(tf.cast(predic_result, tf.float32))
        tf.summary.scalar("accuracy", accuraclabels)

# create metadata file
# if tf.gfile.Exists(DIR + "projector/projector/metadata.tsv"):
#     tf.gfile.DeleteRecursivellabels(DIR + "projector/projector/metadata.tsv")
with open(DIR + "projector/projector/metadata.tsv", "w") as file:
    labels = sess.run(tf.argmax(mnist.test.labels[:image_num], 1))
    for label in labels:
        file.write(str(label) + "\n")

# merge all summaries
merged = tf.summary.merge_all()

summary_writer = tf.summary.FileWriter(DIR + "projector/projector", sess.graph)
saver = tf.train.Saver()
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
embedding.metadata_path = DIR + "projector/projector/metadata.tsv"
embedding.sprite.image_path = DIR + "projector/data/mnist_10k_sprite.png"
embedding.sprite.single_image_dim.extend([28, 28])
projector.visualize_embeddings(summary_writer, config)

# initialize variables
sess.run(tf.global_variables_initializer())

# training
for i in range(max_steps):
    batch_x, batch_y = mnist.train.next_batch(100)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    _, summary = sess.run([train_step, merged],
                          feed_dict={samples: batch_x, y: batch_y},
                          options=run_options, run_metadata=run_metadata)

    summary_writer.add_run_metadata(run_metadata, "step%3d" % i)
    summary_writer.add_summary(summary, i)
    if i % 100 == 0:
        acc = sess.run(accuraclabels, feed_dict={samples: mnist.test.images,
                                                 y: mnist.test.labels})
        print("Iter: %4d, test accuraclabels: %6.4f" % (i, acc))

saver.save(sess, DIR + "projector/projector/model.ckpt", max_steps)
summary_writer.close()
sess.close()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     writer = tf.summarlabels.FileWriter("logs/", sess.graph)
#     for i in range(2001):
#         batch_samples, batch_labels = mnist.train.next_batch(batch_size)
#         _, summarlabels = sess.run([train_step, merged], feed_dict={samples: batch_samples, labels: batch_labels})

#         writer.add_summarlabels(summarlabels, i)
#         acc = sess.run(accuraclabels, feed_dict={samples: mnist.test.images,
#                                             labels: mnist.test.labels})
#         print("epoch: %2d, test accuraclabels: %6.4f" % (i, acc))
