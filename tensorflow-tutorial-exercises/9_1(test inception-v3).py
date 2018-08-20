import tensorflow as tf
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


classId_strId_filepath = "retrain/output_labels.txt"

lines = tf.gfile.GFile(classId_strId_filepath).readlines()
classId_to_strId = {}
for classId, line in enumerate(lines):
    line = line.strip("\n")
    classId_to_strId[classId] = line

def classId_to_depict(classId):
    if classId not in classId_to_strId:
        return ""
    return classId_to_strId[classId]


with tf.Session() as sess:
    # import inception_v3 model into sess.graph
    with tf.gfile.FastGFile("retrain/output_graph.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def)

    softmax_tensor = sess.graph.get_tensor_by_name("import/final_result:0")
    # iterate all images to test
    for root, dirs, files in os.walk("retrain/test_images/"):
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
            for classId in top_5:
                depict = classId_to_depict(classId)
                score = predict[classId]
                print("%s (score = %.5f)" % (depict, score))

            print()

    writer = tf.summary.FileWriter("inception_log/", sess.graph)
    writer.close()
