import tensorflow as tf
import numpy as np
import os
from PIL import Image
from nets import nets_factory


# captcha char_set num
CHAR_SET_NUM = 10
# image height
IMAGE_HEIGHT = 60
# image width
IMAGE_WIDTH = 160
# batch size
BATCH_SIZE = 25
# tfrecord file path
TFRECORD_FILE = "captcha/train.tfrecord"

# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224])
y0 = tf.placeholder(tf.float32, [None])
y1 = tf.placeholder(tf.float32, [None])
y2 = tf.placeholder(tf.float32, [None])
y3 = tf.placeholder(tf.float32, [None])

# learning rate
lr = tf.Variable(0.003, dtype=tf.float32)

# read data from tfrecord file
def read_and_decode(filename):
    # generate queue according to file name
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    # return filename, example file
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           "image": tf.FixedLenFeature([], tf.string),
                                           "label0": tf.FixedLenFeature([], tf.int64),
                                           "label1": tf.FixedLenFeature([], tf.int64),
                                           "label2": tf.FixedLenFeature([], tf.int64),
                                           "label3": tf.FixedLenFeature([], tf.int64),
                                       })
    # fetch image data
    image = tf.decode_raw(features["image"], tf.uint8)
    # image reshape
    image = tf.reshape(image, [224, 224])
    # preprocess image data
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    # get labels
    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)

    return image, label0, label1, label2, label3


# learning rate update
lr_update = tf.assign(lr, lr / 3)

# get training images and corresponding labels
image, label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)

# shuffle batch
image_batch, label0_batch, label1_batch, label2_batch, label3_batch = tf.train.shuffle_batch(
    [image, label0, label1, label2, label3],
    batch_size=BATCH_SIZE,
    capacity=50000,
    min_after_dequeue=10000,
    num_threads=1)

# get network architecture
train_network_fn = nets_factory.get_network_fn(
    "alexnet_v2",
    num_classes=CHAR_SET_NUM,
    weight_decay=.0005,
    is_training=True)

# inputs: a tensorof size: [batch_size, height, width, channel]
X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])

# get output from network
logits0, logits1, logits2, logits3, end_points = train_network_fn(X)

# transfer labels into one_hot format
one_hot_labels0 = tf.one_hot(indices=tf.cast(y0, tf.int32), depth=CHAR_SET_NUM)
one_hot_labels1 = tf.one_hot(indices=tf.cast(y1, tf.int32), depth=CHAR_SET_NUM)
one_hot_labels2 = tf.one_hot(indices=tf.cast(y2, tf.int32), depth=CHAR_SET_NUM)
one_hot_labels3 = tf.one_hot(indices=tf.cast(y3, tf.int32), depth=CHAR_SET_NUM)

# compute loss
loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits0,
    labels=one_hot_labels0))
loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits1,
    labels=one_hot_labels1))
loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits2,
    labels=one_hot_labels2))
loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits3,
    labels=one_hot_labels3))

# define loss and optimizer
total_loss = (loss0 + loss1 + loss2 + loss3) / 4.0
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)

# calculate training accuracy
predic_result0 = tf.equal(tf.argmax(logits0, 1), tf.argmax(one_hot_labels0, 1))
accuracy0 = tf.reduce_mean(tf.cast(predic_result0, tf.float32))

predic_result1 = tf.equal(tf.argmax(logits1, 1), tf.argmax(one_hot_labels1, 1))
accuracy1 = tf.reduce_mean(tf.cast(predic_result1, tf.float32))

predic_result2 = tf.equal(tf.argmax(logits2, 1), tf.argmax(one_hot_labels2, 1))
accuracy2 = tf.reduce_mean(tf.cast(predic_result2, tf.float32))

predic_result3 = tf.equal(tf.argmax(logits3, 1), tf.argmax(one_hot_labels3, 1))
accuracy3 = tf.reduce_mean(tf.cast(predic_result3, tf.float32))

# create saver
saver = tf.train.Saver()

# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(6001):
        # print("iter %d start" % (i))
        # get one batch size image data and labels
        b_image, b_label0, b_label1, b_label2, b_label3 = sess.run(
            [image_batch, label0_batch, label1_batch, label2_batch, label3_batch])
        sess.run(train_step, feed_dict={
            x: b_image, y0: b_label0, y1: b_label1, y2: b_label2, y3: b_label3})

        # print("iter %d start" % (i))
        # calculate loss and accu every 20 iters
        if i % 20 == 0:
            # decrease lr every 2000 iters
            if i % 2000 == 0:
                sess.run(lr_update)

            acc0, acc1, acc2, acc3, loss = sess.run(
                [accuracy0, accuracy1, accuracy2, accuracy3, total_loss],
                feed_dict={x: b_image,
                           y0: b_label0,
                           y1: b_label1,
                           y2: b_label2,
                           y3: b_label3})

            learning_rate = sess.run(lr)
            print("iter: %d, loss: %.3f, training accuracy: %.4f, %.4f, %.4f, %.4f, learning rate: %.4f" % (i, loss, acc0, acc1, acc2, acc3, learning_rate))

            if i == 6000:
                # save trained models
                saver.save(sess, "captcha/model/crack_captcha.model", global_step=i)
                break

    coord.request_stop()
    coord.join(threads)
