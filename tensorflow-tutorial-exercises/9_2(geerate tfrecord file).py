import tensorflow as tf
import os
import sys
import random

# test images num
_NUM_TEST = 500
# random seed
_RADDOM_SEED = 0
# shard num
_NUM_SHARDS = 5
# image set dir
DATASET_DIR = "slim/images"
# labels.txt filepaths
LABELS_FILENAME = "labels.txt"


def _get_tfrecord_filepath(datset_dir, splt_name, shard_id):
    filename = "image_%s_%05d-of-%05d.tfrecord" % (splt_name, shard_id, _NUM_SHARDS)
    tfrecord_filepath = os.path.join(datset_dir, filename)
    return tfrecord_filepath


def _tfrecord_exists(datset_dir):
    for split_name in ["train", "test"]:
        for shard_id in range(_NUM_SHARDS):
            filename = _get_tfrecord_filepath(datset_dir, split_name, shard_id)
            if not tf.gfile.Exists(filename):
                return False

    return True


def _get_filenames_and_classes(datset_dir):
    directories = []
    class_names = []

    for dirs in os.listdir(datset_dir):
        if dirs.startswith('.'):
            continue
        path = os.path.join(datset_dir, dirs)
        if os.path.isdir(path):
            class_names.append(dirs)
            directories.append(path)

    images_filepaths = []

    for directory in directories:
        for path in os.listdir(directory):
            if path.startswith('.'):
                continue
            filename = os.path.join(directory, path)
            images_filepaths.append(filename)

    return images_filepaths, class_names


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def image_to_tfexample(image_data, image_format, class_id):
    # abstract base class for protocol messages
    return tf.train.Example(features=tf.train.Features(feature={
        "image/encoded": bytes_feature(image_data),
        "image/format": bytes_feature(image_format),
        "image/class/label": int64_feature(class_id),
    }))


#  transfer image data into tfrecord format
def _conert_dataset(split_name, image_filepath, class_names_to_ids, dataset_dir):
    assert split_name in ["train", "test"]
    num_image_per_shard = len(image_filepath) // _NUM_SHARDS
    # with tf.Graph().as_default():
    #     with tf.Session() as sess:
    for shard_id in range(_NUM_SHARDS):
        # get tfrecord file path
        filepath = _get_tfrecord_filepath(dataset_dir, split_name, shard_id)
        with tf.python_io.TFRecordWriter(filepath) as tfrecord_writer:
            # start position for every shard
            start_ndx = shard_id * num_image_per_shard
            end_ndx = min((shard_id + 1) * num_image_per_shard, len(image_filepath))
            for i in range(start_ndx, end_ndx):
                try:
                    sys.stdout.write("\r>> Converting image %d/%d shard %d " %
                                     (i + 1, len(image_filepath), shard_id))
                    sys.stdout.flush()
                    # get image data
                    image_data = tf.gfile.FastGFile(image_filepath[i], "rb").read()
                    class_name = os.path.basename(os.path.dirname(image_filepath[i]))
                    class_id = class_names_to_ids[class_name]
                    example = image_to_tfexample(image_data, b" jpg", class_id)
                    tfrecord_writer.write(example.SerializeToString())
                except IOError as e:
                    print("Could not read: ", image_filepath[i])
                    print("Error: ", e)
                    print("Skip it\n")

    sys.stdout.write("\n")
    sys.stdout.flush()


def write_labels_file(labels_to_class_names, datset_dir, filename=LABELS_FILENAME):
    filepath = os.path.join(datset_dir, filename)
    with tf.gfile.Open(filepath, "w") as f:
        for label, name in labels_to_class_names.items():
            f.write("%d:%s\n" % (label, name))


if __name__ == "__main__":
    if _tfrecord_exists(DATASET_DIR):
        print("tfrecord have existed!")
    else:
        # get images filepaths and class names
        images_filepaths, class_names = _get_filenames_and_classes(DATASET_DIR)
        # change class into dic
        class_names_to_ids = dict(zip(class_names, range(len(class_names))))

        # shuffle data into test and train
        random.seed(_RADDOM_SEED)
        random.shuffle(images_filepaths)
        training_filepaths = images_filepaths[_NUM_TEST:]
        testing_filepaths = images_filepaths[:_NUM_TEST]

        # convert data format into tfrecord
        _conert_dataset("train", training_filepaths, class_names_to_ids, DATASET_DIR)
        _conert_dataset("test", testing_filepaths, class_names_to_ids, DATASET_DIR)

        # generate labels.txt file
        labels_to_class_names = dict(zip(range(len(class_names)), class_names))
        write_labels_file(labels_to_class_names, DATASET_DIR)
