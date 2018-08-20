import tensorflow as tf

# x = tf.Variable([1, 2])
# c = tf.constant([3, 3])
# sub = tf.subtract(x, c)
# add = tf.add(x, sub)

# init = tf.global_variables_initializer()

# with tf.Session() as sess:
# 	sess.run(init)
# 	print(sess.run(x))
# 	print(sess.run(sub))
# 	print(sess.run(add))

state = tf.Variable(0, name = 'counter')
updated_value = tf.add(state, 1)
update = tf.assign(state, updated_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	print(sess.run(state))
	for _ in range(5):
		sess.run(update)
		print(sess.run(state))
