'''
Created on Oct 23, 2016

This code implements my MLND capstone algorithm.

@author: pjmartin
'''
import numpy as np
import random as rnd
import scipy as sp
import os
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

# Full CNN architecture for the svhn model.
# Returns a logit value for the length of the house number and
# a logit value for each of the numbers (1-5 positions).
# Xin - svhn image input.
# shape_props - the ShapeProps tuple.
# keep_prob - probability for the dropout layer.
def svhn_model(Xin, var_tuples, shape_props, keep_prob):
	# Conv Layer 1 - 5x5x1x16, ReLU activation.
	l1_vars = var_tuples[0]
	layer1 = tf.nn.relu( tfh.conv2d(Xin, l1_vars[0]) + l1_vars[1], "layer1" )
	maxpool1 = tfh.max_pool_2x2( layer1 ) # Output is (1x16x16x16 tensor)
	# Conv Layer 2 - 5x5x16x32, ReLU activation.
	l2_vars = var_tuples[1]
	# Conv Layer 3 - 5x5x32x64
	l3_vars = var_tuples[2]
	# Fully connected 1 (should connect 8x8x64 to 64)
	# TODO: Try a dropout for better generalization.
	# TODO: Make 2 FC layers: 8x8x64 to 2^11, dropout, 2^11 to 2^6.
	# Return the 6 logit values.
	return maxpool1

def run(train_fname):
	graph = tf.Graph()
	with graph.as_default():
		keep_prob = 0.9
		# Load the training data set.
		train_dataset = load_svhn_pkl(train_fname)
		# Build input pipeline.
		Xplace = tf.placeholder(tf.float32, [None,1024])
		Ximg = tf.reshape(Xplace, [-1,32,32,1], "Ximg")

		# The input placeholder is 6x11 - 6 labels, size 11 one-hot encoding.
		yplace = tf.placeholder(tf.int32, [None, 66])
		# Extract labels by getting ytarget[i] for z_i, where z is an output label,
		# for example, z_L = length of the street number.
		ytarget = tf.reshape(yplace, [-1,6,11], "ytarget")
		
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
# 		z_L = tf.nn.softmax( tf.matmul(dropout, fc_zL_W) + fc_zL_b, name="z_L" )
		# First digit
		fc_z1_W = tf.Variable(tf.truncated_normal([fc1out_size, num_bins], stddev=0.1), name="fc_z1_W")
		fc_z1_b = tf.Variable(tf.constant(0.1,shape=[num_bins]), name="fc_z1_b")
		z_1 = tf.matmul(dropout, fc_z1_W) + fc_z1_b
# 		z_1 = tf.nn.softmax( tf.matmul(dropout, fc_z1_W) + fc_z1_b, name="z_1" )
		# Second digit
		fc_z2_W = tf.Variable(tf.truncated_normal([fc1out_size, num_bins], stddev=0.1), name="fc_z2_W")
		fc_z2_b = tf.Variable(tf.constant(0.1,shape=[num_bins]), name="fc_z2_b")
		z_2 = tf.matmul(dropout, fc_z2_W) + fc_z2_b
# 		z_2 = tf.nn.softmax( tf.matmul(dropout, fc_z2_W) + fc_z2_b, name="z_2" )
		# Third digit
		fc_z3_W = tf.Variable(tf.truncated_normal([fc1out_size, num_bins], stddev=0.1), name="fc_z3_W")
		fc_z3_b = tf.Variable(tf.constant(0.1,shape=[num_bins]), name="fc_z3_b")
		z_3 = tf.matmul(dropout, fc_z3_W) + fc_z3_b
# 		z_3 = tf.nn.softmax( tf.matmul(dropout, fc_z3_W) + fc_z3_b, name="z_3" )
		# Fourth digit
		fc_z4_W = tf.Variable(tf.truncated_normal([fc1out_size, num_bins], stddev=0.1), name="fc_z4_W")
		fc_z4_b = tf.Variable(tf.constant(0.1,shape=[num_bins]), name="fc_z4_b")
		z_4 = tf.matmul(dropout, fc_z4_W) + fc_z4_b
# 		z_4 = tf.nn.softmax( tf.matmul(dropout, fc_z4_W) + fc_z4_b, name="z_4" )
		# Fourth digit
		fc_z5_W = tf.Variable(tf.truncated_normal([fc1out_size, num_bins], stddev=0.1), name="fc_z5_W")
		fc_z5_b = tf.Variable(tf.constant(0.1,shape=[num_bins]), name="fc_z5_b")
		z_5 = tf.matmul(dropout, fc_z5_W) + fc_z5_b
# 		z_5 = tf.nn.softmax( tf.matmul(dropout, fc_z5_W) + fc_z5_b, name="z_5" )
		
		# Build loss operations over the batch.
# 		label_L = ytarget[:,0,:]
# 		xentropy_L = tf.nn.sparse_softmax_cross_entropy_with_logits(z_L, label_L)
		
		# Test Work here:
		sess = tf.Session()
		sess.run(tf.initialize_all_variables())
		Xbatch, ybatch = next_batch(train_dataset["data"], train_dataset["labels"], 1)
		res = sess.run(z_L, feed_dict={Xplace : Xbatch})
		yres = sess.run(ytarget, feed_dict={yplace : ybatch})
		print "Result shape: " + str(np.shape(res))
		
		# Run NN learning processes:
# 		for i in range(num_iters):
			# Learn something! :) 

	return res, yres

if __name__ == '__main__':
	train_fname = '/Users/pjmartin/Documents/Udacity/MachineLearningProgram/Project5/udacity-mle-project5/src/svhn_train.pkl'
	run(train_fname)