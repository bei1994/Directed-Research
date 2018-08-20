import tensorflow as tf

# fetch : carry out multiple ops at the same time.

# input1 = tf.constant(3.0)
# input2 = tf.constant(2.0)
# input3 = tf.constant(1.0)

# add = tf.add(input2, input3)
# mul = tf.multiply(input1, add)

# with tf.Session() as sess:
# 	rel = sess.run([add, mul])
# 	print(rel)


# feed : feed data into placeholder when running.

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
	rel = sess.run(output, feed_dict = {input1: [1,2,3], input2: [3,4,5]})
	print(rel)