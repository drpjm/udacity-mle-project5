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

ShapeProps = namedtuple('ShapeProps', ['filter_size','num_chans','conv_depths','img_size'])

# This function takes a data set and randomly samples a set of size batch_size.
def next_batch(Xs, ys, batch_size):
	Xs_shape = np.shape(Xs)
	num_data = len(Xs)
	idxs = rnd.sample( range(0,num_data), batch_size )
	Xs_sample = Xs[idxs, :] # array of 2D images.
	Xs_sample = Xs_sample.reshape( -1, Xs_shape[1]*Xs_shape[2] )
	ys_sample = ys[idxs, :] # array of 1D vectors.
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
def create_fc_vars(tfgraph, fc_shape, num_outs):
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
def shvn_model(Xin, var_tuples, shape_props, keep_prob):
	Xplace = tf.placeholder(tf.float32, [None,1024])
	Ximg = tf.reshape(Xplace, [-1,32,32,1], "Ximg")
	# Conv Layer 1 - 5x5x1x16, ReLU activation.
	l1_vars = var_tuples[0]
	layer1 = tf.nn.relu( tfh.conv2d(Ximg, l1_vars[0]) + l1_vars[1], "layer1" )
	maxpool1 = tfh.max_pool_2x2( layer1 )
	# Conv Layer 2 - 5x5x16x32, ReLU activation.
	l2_vars = var_tuples[1]
	# Conv Layer 3 - 5x5x32x64
	l3_vars = var_tuples[2]
	# Fully connected 1 (should connect 8x8x64 to 64)
	# TODO: Try a dropout for better generalization.
	# TODO: Make 2 FC layers: 8x8x64 to 2^11, dropout, 2^11 to 2^6.
	# Return the 6 logit values.
	return

def run():
	graph = tf.Graph()
	# Setup tensor shape properties.
	svhn_props = ShapeProps(5, 1, [16,32,64], 32)
	# Create conv layer variables
	vars = create_convvars(svhn_props)
	
	# Build model
	
	# Init variables
	
	# Run

	return
