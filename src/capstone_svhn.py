'''
Created on Oct 23, 2016

This code implements my MLND capstone algorithm.

@author: pjmartin
'''
import numpy as np
import random as rnd
import scipy as sp
import os
import time
import tfhelpers as tfh
import tensorflow as tf
from collections import namedtuple
from svhn_preprocess import load_svhn_pkl
from math import pow

ShapeProps = namedtuple('ShapeProps', ['filter_size','num_chans','conv_depths','img_size'])

# This function takes a data set and randomly samples a set of size batch_size.
def next_batch(Xs, ys, batch_size):
	Xs_shape = np.shape(Xs)
	ys_shape = np.shape(ys)
	num_data = len(Xs)
	idxs = rnd.sample( range(0,num_data), batch_size )
	Xs_sample = Xs[idxs, :] # array of 2D images.
	Xs_sample = Xs_sample.reshape( -1, Xs_shape[1]*Xs_shape[2] )
	ys_sample = ys[idxs, :] # array of 1D vectors.
	ys_sample = ys_sample.reshape(-1, ys_shape[1]*ys_shape[2])
	return Xs_sample, ys_sample

def test_next_batch(fname, size):
	# Used for Eclipse PyDev development/test:
	# /Users/pjmartin/Documents/Udacity/MachineLearningProgram/Project5/udacity-mle-project5/src/svhn_train.pkl
	train_dataset = load_svhn_pkl(fname)
	return next_batch(train_dataset["data"], train_dataset["labels"], size)

# Creates the weight and bias variables used by the model.
def create_convvars(shape_props):
	depth1 = shape_props.conv_depths[0]
	depth2 = shape_props.conv_depths[1]
	depth3 = shape_props.conv_depths[2]
	fsize = shape_props.filter_size
	nchans = shape_props.num_chans

	# conv1 variables
	W1 = tfh.weight_variable([fsize, fsize, nchans, depth1], "conv1_W")
	b1 = tfh.bias_variable([depth1], "conv1_b")
	# conv2 variables
	W2 = tfh.weight_variable([fsize, fsize, depth1, depth2], "conv2_W")
	b2 = tfh.bias_variable([depth2], "conv2_b")
	# conv3 variables
	W3 = tfh.weight_variable([fsize, fsize, depth2, depth3], "conv3_W")
	b3 = tfh.bias_variable([depth3], "conv3_b")
	return [ (W1, b1), (W2, b2), (W3, b3) ]

# Creates FC layer variables for each of the num_outs required.
def create_fc_vars(fc_shape, num_outs):
	fc_var_tuples = []
	for i in range(1,num_outs+1):
		W = tfh.weight_variable(fc_shape, "Wfc" + str(i))
		b = tfh.bias_variable([ fc_shape[1] ], "bfc" + str(i))
		fc_var_tuples.append( (W,b) )
	return fc_var_tuples

def run(train_fname, num_steps, isTest):
	graph = tf.Graph()
	with graph.as_default():
		keep_prob = 0.8
		# Load the training data set.
		train_dataset = load_svhn_pkl(train_fname)
		# Build input pipeline.
		Xplace = tf.placeholder(tf.float32, [None,1024])
		Ximg = tf.reshape(Xplace, [-1,32,32,1], "Ximg")

		# The input placeholder is 6x11 - 6 labels, size 11 one-hot encoding.
		y_place = tf.placeholder(tf.float32, [None, 66])
# 		y_place = tf.placeholder(tf.int32, [None, 66])
		# Extract labels by getting y_target[i] for z_i, where z is an output label,
		# for example, z_L = length of the street number.
		y_target = tf.reshape(y_place, [-1,6,11], "y_target")
		
		# Setup tensor shape properties.
		svhn_props = ShapeProps(5, 1, [16,32,64], 32)
		# Create layer variables (conv then FC)
		var_tuples = create_convvars(svhn_props)
		
		#
		# Build model operations
		#
		# Layer 1: Conv -> Relu -> Maxpool; Output is 1x16x16x16 tensor.
		l1_vars = var_tuples[0]
		layer1 = tf.nn.relu( tfh.conv2d(Ximg, l1_vars[0]) + l1_vars[1], "layer1" )
		maxpool1 = tfh.max_pool_2x2( layer1 ) # 
		# Layer 2: Conv -> Relu -> Maxpool; Output is 1x8x8x32 tensor.
		l2_vars = var_tuples[1]
		layer2 = tf.nn.relu( tfh.conv2d(maxpool1, l2_vars[0]) + l2_vars[1], "layer2" )
		maxpool2 = tfh.max_pool_2x2( layer2 )
		# Layer 3: Conv -> Relu; Output is a 1x8x8x64 tensor.
		l3_vars = var_tuples[2]
		layer3 = tf.nn.relu( tfh.conv2d(maxpool2, l3_vars[0]) + l3_vars[1], "layer3" )
		# Fully Connected layer: Creates a feature vector of size 2^11 that will be
		# fed into the 6 different feature extractors.
		fc1in_size = int(layer3.get_shape()[1]*layer3.get_shape()[2]*layer3.get_shape()[3])
		fc1out_size = int(pow(2,11))
		fc1_W = tf.Variable(tf.truncated_normal([fc1in_size, fc1out_size], stddev=0.1), name="fc1_W")
		fc1_b = tf.Variable(tf.constant(0.1,shape=[fc1out_size]), name="fc1_b")
		fc1_input = tf.reshape(layer3, [-1,fc1in_size], "fc1_input1")
		fc1_layer = tf.nn.relu( tf.matmul(fc1_input, fc1_W) + fc1_b, name="fc1_layer" )
		# Add a dropout layer before the feature vector is sent to the variables.
		dropout = tf.nn.dropout(fc1_layer, keep_prob)		
		#
		# The output of the fully connected layer is fed into 6 different
		# variable outputs: FC -> softmax.
		#
		num_bins = 11
		# Length
		fc_zL_W = tf.Variable(tf.truncated_normal([fc1out_size, num_bins], stddev=0.1), name="fc_zL_W")
		fc_zL_b = tf.Variable(tf.constant(0.1,shape=[num_bins]), name="fc_zL_b")
		z_L = tf.matmul(dropout, fc_zL_W) + fc_zL_b

		# First digit
		fc_z1_W = tf.Variable(tf.truncated_normal([fc1out_size, num_bins], stddev=0.1), name="fc_z1_W")
		fc_z1_b = tf.Variable(tf.constant(0.1,shape=[num_bins]), name="fc_z1_b")
		z_1 = tf.matmul(dropout, fc_z1_W) + fc_z1_b

		# Second digit
		fc_z2_W = tf.Variable(tf.truncated_normal([fc1out_size, num_bins], stddev=0.1), name="fc_z2_W")
		fc_z2_b = tf.Variable(tf.constant(0.1,shape=[num_bins]), name="fc_z2_b")
		z_2 = tf.matmul(dropout, fc_z2_W) + fc_z2_b

		# Third digit
		fc_z3_W = tf.Variable(tf.truncated_normal([fc1out_size, num_bins], stddev=0.1), name="fc_z3_W")
		fc_z3_b = tf.Variable(tf.constant(0.1,shape=[num_bins]), name="fc_z3_b")
		z_3 = tf.matmul(dropout, fc_z3_W) + fc_z3_b

		# Fourth digit
		fc_z4_W = tf.Variable(tf.truncated_normal([fc1out_size, num_bins], stddev=0.1), name="fc_z4_W")
		fc_z4_b = tf.Variable(tf.constant(0.1,shape=[num_bins]), name="fc_z4_b")
		z_4 = tf.matmul(dropout, fc_z4_W) + fc_z4_b
# 		
		# Fifth digit
		fc_z5_W = tf.Variable(tf.truncated_normal([fc1out_size, num_bins], stddev=0.1), name="fc_z5_W")
		fc_z5_b = tf.Variable(tf.constant(0.1,shape=[num_bins]), name="fc_z5_b")
		z_5 = tf.matmul(dropout, fc_z5_W) + fc_z5_b
# 		
		#
		# Build loss operations over the batch.
		#
		mean_L = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(z_L, y_target[:,0,:]) )
		mean_1 = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(z_L, y_target[:,1,:]) )
		mean_2 = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(z_L, y_target[:,2,:]) )
		mean_3 = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(z_L, y_target[:,3,:]) )
		mean_4 = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(z_L, y_target[:,4,:]) )
		mean_5 = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(z_L, y_target[:,5,:]) )
		total_loss = mean_L + mean_1 + mean_2 + mean_3 + mean_4 + mean_5
		# Attach the Adagrad optimizer with a decaying learning rate.
		step_idx = tf.Variable(0)
		learn_rate = tf.train.exponential_decay(0.1, step_idx, num_steps, 0.95)
		adagrad_opt = tf.train.AdagradOptimizer( learn_rate, name="train_adagrad_with_decay" )
		train_step = adagrad_opt.minimize(total_loss, global_step=step_idx)
		
		y_L = tf.nn.softmax(z_L)
		y_1 = tf.nn.softmax(z_1)
		y_2 = tf.nn.softmax(z_2)
		y_3 = tf.nn.softmax(z_3)
		y_4 = tf.nn.softmax(z_4)
		y_5 = tf.nn.softmax(z_5)
		# Probability (y_probs) of label. Tensor shape = (batch_size, 6, 11)
		y_probs = tf.pack([y_L, y_1, y_2, y_3, y_4, y_5], axis=1, name="y_probs")
		# Argmax tensor shape = (batch_size, 6)
		y_argmax = tf.arg_max(y_probs, dimension=2)
		# Turn into a one hot encoded tensor with size (batch_size, 6, 11)
		y_pred = tf.one_hot(y_argmax, depth=num_bins, on_value=1, off_value=0, axis=-1, dtype=tf.int32, name="y_pred")
		
		if isTest:
			# Test Work here:
			sess = tf.Session()
			sess.run(tf.initialize_all_variables())
			Xbatch, ybatch = next_batch(train_dataset["data"], train_dataset["labels"], 2)
			loss, preds, targets = sess.run([total_loss, y_pred, y_target], feed_dict={Xplace : Xbatch, y_place : ybatch})
			return loss, preds, targets
		else:
			# Run NN learning processes:
			sess = tf.Session()
			sess.run(tf.initialize_all_variables())
			batch_size = 1
# 			runtimes = []
	 		for i in range(num_steps):
				Xbatch, ybatch = next_batch(train_dataset["data"], train_dataset["labels"], batch_size)
				_, curr_probs, curr_loss  = sess.run([train_step, y_probs, total_loss], feed_dict={Xplace : Xbatch, y_place : ybatch})
				
				if i % 250 == 0:
# 					runtimes.append(end-start)
					print "Step " + str(i) + ": current loss = " + str(curr_loss)
					# TODO: Compute the accuracy
					
					
# 			avg_runtimes = np.mean(np.array(runtimes))
# 			print "Average runtime = " + str(avg_runtimes)
			return

# if __name__ == '__main__':
# 	train_fname = '/Users/pjmartin/Documents/Udacity/MachineLearningProgram/Project5/udacity-mle-project5/src/svhn_train.pkl'
# 	run(train_fname, 501, True)