import tensorflow as tf
import os
import tarfile
import requests


# pretrained model download website
inception_pretrain_model_url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"

# model store directory
inception_pretrain_model_dir = "inception_model/inception_v3_2015_12_05"
if not os.path.exists(inception_pretrain_model_dir):
    os.makedirs(inception_pretrain_model_dir)

# get file name and file path
filename = inception_pretrain_model_url.split("/")[-1]
filepath = os.path.join(inception_pretrain_model_dir, filename)


# download inception_v3_2015_12_05
if not os.path.exists(filepath):
    print("download: ", filename)
    r = requests.get(inception_pretrain_model_url, stream=True)
    with open(filepath, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
print("finish: ", filename)

# extract .tgz file
tarfile.open(filepath, "r:gz").extractall(inception_pretrain_model_dir)

# model store dir
log_dir = "inception_log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# classify_graph_def.pb is pretrained model
inception_graph_def_file = os.path.join(inception_pretrain_model_dir, "classify_image_graph_def.pb")
with tf.Session() as sess:
    # create a graph to store pretrained models
    with tf.gfile.FastGFile(inception_graph_def_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def)

    # save graph model
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    writer.close()
