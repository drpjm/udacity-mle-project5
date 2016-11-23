'''
Created on Sep 23, 2016

@author: pjmartin
'''
import tensorflow as tf
import numpy as np
import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt
import os
import sys
import cPickle as pickle
import sklearn.preprocessing as skproc
import tfhelpers as tfh
import random as rnd

# Function that will create a randomly sampled batch of data.
def next_batch(xs, ys, batch_size):
	xs_shape = np.shape(xs)
# 	ys_shape = np.shape(ys)
	idxs = rnd.sample(range(0,xs_shape[0]), batch_size)
	xs_rand = xs[idxs, :]
	# Turn input image (NxN) a single array of size N*N.
	xs_rand = xs_rand.reshape(-1, xs_shape[1]*xs_shape[2])
	ys_rand = ys[idxs,:]
	return xs_rand, ys_rand

def load_all_data(file_str):
	curr_dir = os.getcwd()
	data_file = open(os.path.join(curr_dir, file_str))
	nonMnist_all = pickle.load(data_file)
	# Set up the one-hot encoding...
	train_ys = nonMnist_all['train_labels'].reshape(-1,1)
	test_ys = nonMnist_all['test_labels'].reshape(-1,1)
	enc = skproc.OneHotEncoder(sparse=False)
	enc.fit(test_ys)
	train_data = {'x': nonMnist_all['train_dataset'], 'y' : enc.transform(train_ys)}
	test_data = {'x' : nonMnist_all['test_dataset'], 'y' : enc.transform(test_ys)}
	return train_data, test_data

def run_nonmnist():
	print 'Running nonMNIST CNN task.'
	print 'Load the nonMNIST data...'
# 	For running at command line in the src directory.
# 	train_data, test_data = load_all_data('notMNIST_sanitized.pickle')
	train_data, test_data = load_all_data('/Users/pjmartin/Documents/Udacity/MachineLearningProgram/Project5/udacity-mle-project5/src/notMNIST_sanitized.pickle')
	'''
		Build up the TF basics...
	'''
	cnn_sess = tf.Session()
# 	# input - 28x28 reshaped into a
	x = tf.placeholder(tf.float32, [None,784])
# 	# output labels - one hot vectors corresponding to leters, 'A', 'B', etc.
	y = tf.placeholder(tf.float32, [None,10])
	ximg = tf.reshape(x, [-1,28,28,1], 'ximg')

	# Layer 1: 2DConv -> ReLU -> 2x2 max pool
	Wc1 = tfh.weight_variable([5,5,1,32])
	bc1 = tfh.bias_variable([32])
	tfh.variable_summaries(Wc1, 'Wc1')
	tfh.variable_summaries(bc1, 'bc1')
	layer1_out = tfh.max_pool_2x2( tf.nn.relu(tfh.conv2d(ximg, Wc1) + bc1, 'layer1_out') )
	# output is now 28 / 2 = 14

	# Layer 2: Layer1 Output -> 2DConv -> ReLU -> 2x2 max pool
	Wc2 = tfh.weight_variable([5,5,32,64])
	bc2 = tfh.bias_variable([64])
	tfh.variable_summaries(Wc2, 'Wc2')
	tfh.variable_summaries(bc2, 'bc2')
	layer2_out = tfh.max_pool_2x2( tf.nn.relu(tfh.conv2d(layer1_out, Wc2) + bc2, 'layer2_out') )
	# output is now 14 / 2 = 7

	# First fully connected layer: using 7x7x64 features.
# 	fullyconn1_out = tfh.fully_conn_nn_layer(tf.reshape(layer2_out,[-1,3136]), 3136, 1024, 'fc1')
	Wfc1 = tfh.weight_variable([3136, 1024])
	bfc1 = tfh.bias_variable([1024])
	layer2_out_flat = tf.reshape(layer2_out,[-1,3136])
	fullyconn1_out = tf.nn.relu(tf.matmul(layer2_out_flat, Wfc1) + bfc1)

	# Just like MNIST, add a dropout
	keep_prob = tf.placeholder(tf.float32)
	dropped_out = tf.nn.dropout(fullyconn1_out, keep_prob, name='dropout1')

	# Connect the output from the dropout layer to the final, softmax fully conn layer.
	Wfc2 = tfh.weight_variable([1024, 10])
	bfc2 = tfh.bias_variable([10])
	y_nn = tf.nn.softmax(tf.matmul(dropped_out, Wfc2) + bfc2)
# 	y_nn = tfh.fully_conn_nn_layer(dropped_out, 1024, 10, 'fc2', act_fn=tf.nn.softmax)

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_nn), reduction_indices=[1]))
	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	with tf.name_scope('performance'):
		with tf.name_scope('correct_prediction'):
			correct_pred = tf.equal(tf.arg_max(y_nn, 1), tf.arg_max(y, 1))
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		tf.scalar_summary('accuracy', accuracy)
	cnn_sess.run(tf.initialize_all_variables())

	merged_summaries = tf.merge_all_summaries()
	summaries_loc = os.path.join('/Users/pjmartin/Documents/Udacity/MachineLearningProgram/Project5/udacity-mle-project5/src', 'summaries', 'non_mnist')
	train_writer = tf.train.SummaryWriter(summaries_loc + '/train', cnn_sess.graph)

	for i in range(1000):
		xbatch, ybatch = next_batch(train_data['x'], train_data['y'], 75)
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x: xbatch, y: ybatch, keep_prob: 1.0}, session=cnn_sess)
			print("step %d, training accuracy %g"%(i, train_accuracy))
		else:
			summary, _ = cnn_sess.run([merged_summaries, train_step], feed_dict={x:xbatch, y: ybatch, keep_prob: 0.5})
			train_writer.add_summary(summary, i)

	cnn_sess.close()
# I need the following for pydev interactive session path setting.
# sys.path.append(os.path.join('/Users/pjmartin/Documents/Udacity/MachineLearningProgram/Project5/udacity-mle-project5/src'))

# if __name__ == '__main__':
# 	run_nonmnist()
