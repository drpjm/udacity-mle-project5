'''
Created on Sep 23, 2016

@author: pjmartin
'''

import numpy as np
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
import random as rnd
import os
from tfhelpers import fully_conn_nn_layer
from tfhelpers import variable_summaries

# Function that will create a randomly sampled batch of data.
def next_batch(xs, ys, batch_size):
	xs_shape = np.shape(xs)
# 	ys_shape = np.shape(ys)
	idxs = rnd.sample(range(0,xs_shape[0]), batch_size)
	xs_rand = xs[idxs, :]
	ys_rand = ys[idxs,:]
	return xs_rand, ys_rand 

def run_toy():
	print 'Running toy NN example...'
	# Build summary output location.
	curr_dir = os.getcwd()
	summaries_loc = os.path.join(curr_dir, 'summaries', 'toy')
	
	# Create the data set from the example, but tweak it.
	N = 300
	Ntest = 50
	D = 2
	K = 3
	X = np.zeros((N*K, D)) # row is a single vector (x1, x2)
	y = np.zeros(N*K, dtype='uint8')
	y_onehot = np.zeros((N*K, K))
	Xtest = np.zeros((Ntest*K, D))
	ytest = np.zeros(Ntest*K, dtype='uint8')
	y_onehot_test = np.zeros((Ntest*K, K))
	
	for i in xrange(K):
		xidx = range(N*i,N*(i+1))
		xt_idx = range(Ntest*i,Ntest*(i+1))
		r = np.linspace(0.0,1,N)
		theta = np.linspace(i*4, (i+1)*4, N) + np.random.randn(N)*0.15
		X[xidx] = np.c_[np.float32(r*np.sin(theta)), np.float32(r*np.cos(theta))]
		y[xidx] = i
		class_as_onehot = np.zeros(K)
		class_as_onehot[i] = 1
		y_onehot[xidx] = class_as_onehot
		r_test = np.linspace(0.0,1,Ntest)
		theta_test = np.linspace(i*4, (i+1)*4, Ntest) + np.random.randn(Ntest)*0.2
		Xtest[xt_idx] = np.c_[r_test*np.sin(theta_test), r_test*np.cos(theta_test)]
		ytest[xt_idx] = i
		class_as_onehot_test = np.zeros(K)
		class_as_onehot_test[i] = 1
		y_onehot_test[xt_idx] = class_as_onehot_test
	
	# Create the TensorFlow structure.
	# TF placeholder for the input and target data (y_targ)
	x = tf.placeholder(tf.float32, [None, 2])
	y_targ = tf.placeholder(tf.float32, [None, K])
	num_neurons = 100
	
	# Layer 1 - ReLU activation layer
	layer1, W1, b1 = fully_conn_nn_layer(x, 2, num_neurons, 'layer1')
	
	# Layer 2 - Softmax to generate class probabilities
	y_output, W2, b2 = fully_conn_nn_layer(layer1, num_neurons, K, 'layer2', act_fn=tf.nn.softmax)
	
	# Loss function construction:
	l = 0.001
	reg_loss = 0.5 * l * (tf.reduce_sum(W1*W1) + tf.reduce_sum(W2*W2)) 
	xentropy = tf.reduce_mean(-tf.reduce_sum( y_targ * tf.log(y_output), reduction_indices=[1] ))
	loss = xentropy + reg_loss
	
	# Create the training step within name scope for debug.
	with tf.name_scope('train'):
		train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	
	# Fire up TF!
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	
	# Result summary setup:
	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_pred = tf.equal(tf.argmax(y_output,1), tf.argmax(y_targ,1))
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		tf.scalar_summary('accuracy', accuracy)
	
	# Send out the summaries for TensorBoard.
	merged_summaries = tf.merge_all_summaries()
	train_writer = tf.train.SummaryWriter(summaries_loc + '/train', sess.graph)
	test_writer = tf.train.SummaryWriter(summaries_loc + '/test')
	
	for i in range(1000):
		batch_xs, batch_ys = next_batch(X, y_onehot, 75)
		sess.run(train_step, feed_dict={x: batch_xs, y_targ: batch_ys})
		if i % 10 == 0:
			summary, acc = sess.run([merged_summaries, accuracy], feed_dict={x: Xtest, y_targ: y_onehot_test})
			test_writer.add_summary(summary, i)
			print('Accuracy at step %s: %s' % (i, acc))
		else:
			summary, _ = sess.run([merged_summaries, train_step], feed_dict={x: batch_xs, y_targ: batch_ys})
			train_writer.add_summary(summary, i)

if __name__ == '__main__':
	run_toy()