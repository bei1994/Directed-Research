import tensorflow as tf
import numpy as np
import os
from PIL import Image
from nets import nets_factory
import matplotlib.pyplot as plt

# captcha char_set num
CHAR_SET_NUM = 10
# image height
IMAGE_HEIGHT = 60
# image width
IMAGE_WIDTH = 160
# batch size
BATCH_SIZE = 1
# tfrecord file path
TFRECORD_FILE = "captcha/test.tfrecord"


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
    image_raw = tf.reshape(image, [224, 224])
    # preprocess image data
    image = tf.reshape(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    # get labels
    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)

    return image, image_raw, label0, label1, label2, label3


# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224])

# inputs: a tensorof size: [batch_size, height, width, channel]
X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])


# get training images and corresponding labels
image, image_raw, label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)

# shuffle batch
image_batch, image_raw_batch, label0_batch, label1_batch, label2_batch, label3_batch = tf.train.shuffle_batch(
    [image, image_raw, label0, label1, label2, label3],
    batch_size=BATCH_SIZE,
    capacity=50000,
    min_after_dequeue=10000,
    num_threads=1)

# get network architecture
train_network_fn = nets_factory.get_network_fn(
    "alexnet_v2",
    num_classes=CHAR_SET_NUM,
    weight_decay=.0005,
    is_training=False)


# get output from network
logits0, logits1, logits2, logits3, end_points = train_network_fn(X)

# calculate testing accuracy
predic_result0 = tf.reshape(logits0, [-1, CHAR_SET_NUM])
predic_result0 = tf.argmax(predic_result0, 1)

predic_result1 = tf.reshape(logits1, [-1, CHAR_SET_NUM])
predic_result1 = tf.argmax(predic_result1, 1)

predic_result2 = tf.reshape(logits2, [-1, CHAR_SET_NUM])
predic_result2 = tf.argmax(predic_result2, 1)

predic_result3 = tf.reshape(logits3, [-1, CHAR_SET_NUM])
predic_result3 = tf.argmax(predic_result3, 1)

# create saver
saver = tf.train.Saver()

# testing
with tf.Session() as sess:

    writer = tf.summary.FileWriter("captcha/logs/", sess.graph)
    saver.restore(sess, 'captcha/model/crack_captcha.model-6000')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(10):
        # get one batch size image data and labels
        b_image, b_image_raw, b_label0, b_label1, b_label2, b_label3 = sess.run([
                                                                                image_batch,
                                                                                image_raw_batch,
                                                                                label0_batch,
                                                                                label1_batch,
                                                                                label2_batch,
                                                                                label3_batch])
        # display raw images
        img = Image.fromarray(b_image_raw[0], 'L')
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        # print labels
        print('label: ', b_label0, b_label1, b_label2, b_label3)
        # predicted results
        label0, label1, label2, label3 = sess.run([predic_result0,
                                                   predic_result1,
                                                   predic_result2,
                                                   predic_result3], feed_dict={x: b_image})
        # print predicted labels
        print('predict: ', label0, label1, label2, label3)

    coord.request_stop()
    coord.join(threads)
