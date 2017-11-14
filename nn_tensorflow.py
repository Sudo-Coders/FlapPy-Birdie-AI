import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler

train = pd.read_csv('Nitin_130.csv')

X_data = pd.DataFrame()
X_data['X'] = train['X']
X_data['Y'] = train['Y']
y_data = train['click']

# oversampling of data
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X_data, y_data)

# print X_resampled[0, :]

X = tf.placeholder(tf.float32, [None, 2])
y = tf.placeholder(tf.float32, [None, 1])

middle = 6

w_1 = tf.Variable(tf.truncated_normal([2, middle]))
b_1 = tf.Variable(tf.truncated_normal([1, middle]))
w_2 = tf.Variable(tf.truncated_normal([middle, 1]))
b_2 = tf.Variable(tf.truncated_normal([1, 1]))

def sigmoid(x):
	return tf.div(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))


def sigmoidPrime(x):
	return tf.multiply(sigmoid(x), tf.subtract(tf.constant(1.0), sigmoid(x)))



with tf.device('/cpu:0'):
	# Forward Propogation
	z_1 = tf.add(tf.matmul(X, w_1), b_1)
	a_1 = tf.sigmoid(z_1)
	z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
	a_2 = tf.sigmoid(z_2)


	# Backpropogation Algorithm
	diff = tf.subtract(a_2, y)

	d_z_2 = tf.multiply(diff, sigmoidPrime(z_2))
	d_b_2 = d_z_2
	d_w_2 = tf.matmul(tf.transpose(a_1), d_z_2)

	d_a_1 = tf.matmul(d_z_2, tf.transpose(w_2))
	d_z_1 = tf.multiply(d_a_1, sigmoidPrime(z_1))
	d_b_1 = d_z_1
	d_w_1 = tf.matmul(tf.transpose(X), d_z_1)

	eta = tf.constant(0.5)
	step = [tf.assign(w_1, tf.subtract(w_1, tf.multiply(eta, d_w_1))), tf.assign(b_1, tf.subtract(b_1, tf.multiply(eta, tf.reduce_mean(d_b_1, axis=[0])))), tf.assign(w_2, tf.subtract(w_2, tf.multiply(eta, d_w_2))), tf.assign(b_2, tf.subtract(b_2, tf.multiply(eta, tf.reduce_mean(d_b_2, axis=[0]))))]

	acct_mat = tf.equal(tf.argmax(a_2, 1), tf.argmax(y, 1))
	acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in xrange(180000):
	res = sess.run(acct_res, feed_dict={X : X_resampled[i, :].reshape(1, 2), y : y_resampled[i].reshape(1, 1)})
	if i%1000 == 0:
		print "Hello", res

saver = tf.train.Saver()
save_path = saver.save(sess, 'NN_model.ckpt')
print('Model is  successfully trained')