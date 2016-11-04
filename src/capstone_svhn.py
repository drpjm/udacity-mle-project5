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
from svhn_preprocess import load_svhn_pkl

# This function takes a data set and randomly samples a set of size batch_size.
def next_batch(Xs, ys, batch_size):
	num_data = len(Xs)
	idxs = rnd.sample(range(num_data), batch_size)
	Xs_sample = Xs[idxs, :, :] # array of 2D images.
	ys_sample = ys[idxs, :] # array of 1D vectors.
	return Xs_sample, ys_sample

def test_next_batch(fname, size):
	# Used for Eclipse PyDev development/test:
	# /Users/pjmartin/Documents/Udacity/MachineLearningProgram/Project5/udacity-mle-project5/src/svhn_train.pkl
	train_dataset = load_svhn_pkl(fname)
	return next_batch(train_dataset["data"], train_dataset["labels"], size)

def build_model():
	return