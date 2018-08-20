import tensorflow as tf
import os
import sys
import random
from PIL import Image
import numpy as np

# test images num
_NUM_TEST = 500
# random seed
_RADDOM_SEED = 0
# imageset filepath
DATASET_DIR = "captcha/images"
# tfrecord store filepaths
TFRECORD_DIR = "captcha/"

# check if tfrecord file exists


def _tfrecord_exists(datset_dir):
    for split_name in ["train", "test"]:
        filename = os.path.join(datset_dir, split_name + ".tfrecord")
        if not tf.gfile.Exists(filename):
            return False

    return True

# get all images files


def _get_filenames_and_classes(datset_dir):
    images_filepaths = []

    for filename in os.listdir(datset_dir):
        if filename.startswith('.'):
            continue
        filepath = os.path.join(datset_dir, filename)
        images_filepaths.append(filepath)

    return images_filepaths


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def image_to_tfexample(image_data, label0, label1, label2, label3):
    # abstract base class for protocol messages
    return tf.train.Example(features=tf.train.Features(feature={
        "image": bytes_feature(image_data),
        "label0": int64_feature(label0),
        "label1": int64_feature(label1),
        "label2": int64_feature(label2),
        "label3": int64_feature(label3),
    }))


#  transfer image data into tfrecord format
def _conert_dataset(split_name, image_filepath, dataset_dir):
    assert split_name in ["train", "test"]

    filepath = os.path.join(dataset_dir, split_name + '.tfrecord')
    with tf.python_io.TFRecordWriter(filepath) as tfrecord_writer:
        for i, filename in enumerate(image_filepath):
            try:
                sys.stdout.write("\r>> Converting image %d/%d" %
                                 (i + 1, len(image_filepath)))
                sys.stdout.flush()
                # get image data
                image_data = Image.open(filename)
                image_data = image_data.resize((224, 224))
                image_data = np.array(image_data.convert('L'))
                image_data = image_data.tobytes()
                # get labels
                labels = filename.split('/')[-1][0:4]
                num_lables = [int(labels[i]) for i in range(4)]

                # get protobuf data
                example = image_to_tfexample(image_data, num_lables[0], num_lables[1], num_lables[2], num_lables[3])
                tfrecord_writer.write(example.SerializeToString())
            except IOError as e:
                print("Could not read: ", image_filepath[i])
                print("Error: ", e)
                print("Skip it\n")

    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    if _tfrecord_exists(TFRECORD_DIR):
        print("tfrecord have existed!")
    else:
        # get images filepaths
        images_filepaths = _get_filenames_and_classes(DATASET_DIR)

        # shuffle data into test and train
        random.seed(_RADDOM_SEED)
        random.shuffle(images_filepaths)
        training_filepaths = images_filepaths[_NUM_TEST:]
        testing_filepaths = images_filepaths[:_NUM_TEST]

        # convert image_data into tfrecord
        _conert_dataset("train", training_filepaths, TFRECORD_DIR)
        _conert_dataset("test", testing_filepaths, TFRECORD_DIR)
        print("tfrecord file generated!!")
