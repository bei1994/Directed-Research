import tensorflow as tf
import sys

print(sys.executable)
print(sys.version)

m1 = tf.constant([[3, 3]])
m2 = tf.constant([[2], [3]])
prod = tf.matmul(m1, m2)


with tf.Session() as sess:
    # result = sess.run(prod)
    print(sess.run(m1))
    print(sess.run(m2))
    print(sess.run(prod))
    print(m1)
