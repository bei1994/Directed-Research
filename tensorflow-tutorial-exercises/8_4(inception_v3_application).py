import tensorflow as tf
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class NodeLookup(object):
    def __init__(self):
        classId_strId_filepath = "inception_model/inception_v3_2015_12_05/imagenet_2012_challenge_label_map_proto.pbtxt"
        strId_depict_filepath = "inception_model/inception_v3_2015_12_05/imagenet_synset_to_human_label_map.txt"
        self.lookup_table = self.load(classId_strId_filepath, strId_depict_filepath)

    def load(self, classId_strId_filepath, strId_depict_filepath):
        # read classId_strId file and create dict: {classId:strId}
        lines = tf.gfile.GFile(classId_strId_filepath).readlines()
        classId_to_strId = {}
        for line in lines:
            if line.startswith("  target_class:"):
                classId = int(line.split(": ")[1])
            if line.startswith("  target_class_string:"):
                classId_to_strId[classId] = line.split(": ")[1][1:-2]

        # read strId_depict file and create dict: {strId:depict}
        lines = tf.gfile.GFile(strId_depict_filepath).readlines()
        strId_to_depict = {}
        for line in lines:
            line = line.strip("\n")
            parsed = line.split("\t")
            strId = parsed[0]
            depict = parsed[1]
            strId_to_depict[strId] = depict

        # build classId_to_depict dict: {classId:depict}
        classId_to_depict = {}
        for k, v in classId_to_strId.items():
            classId_to_depict[k] = strId_to_depict[v]

        return classId_to_depict

    def classId_to_depict(self, classId):
        if classId not in self.lookup_table:
            return ""
        return self.lookup_table[classId]


with tf.Session() as sess:
    # import inception_v3 model into sess.graph
    with tf.gfile.FastGFile("inception_model/inception_v3_2015_12_05/classify_image_graph_def.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def)

    softmax_tensor = sess.graph.get_tensor_by_name("import/softmax:0")
    # iterate all images to test
    for root, dirs, files in os.walk("images/"):
        for file in files:
            if not file.endswith('.jpg') or file.startswith('.'):
                continue  # Skip!
            # fetch image
            image_data = tf.gfile.FastGFile(os.path.join(root, file), "rb").read()
            predict = sess.run(softmax_tensor, feed_dict={"import/DecodeJpeg/contents:0": image_data})
            predict = np.squeeze(predict)

            image_path = os.path.join(root, file)
            print(image_path)
            # img = Image.open(image_path)
            # plt.imshow(img)
            # plt.axis("off")
            # plt.show()

            top_5 = predict.argsort()[-5:][::-1]
            node = NodeLookup()
            for classId in top_5:
                depict = node.classId_to_depict(classId)
                score = predict[classId]
                print("%s (score = %.5f)" % (depict, score))

            print()
