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

def load_all_data(file_str):
	curr_dir = os.getcwd()
	data_file = open(os.path.join(curr_dir, file_str))
	nonMnist_all = pickle.load(data_file)
	# Set up the one-hot encoding...
	train_ys = nonMnist_all['train_labels'].reshape(-1,1)
	test_ys = nonMnist_all['test_labels'].reshape(-1,1)
	enc = skproc.OneHotEncoder()
	enc.fit(test_ys)
	train_data = {'x': nonMnist_all['train_dataset'], 'y' : enc.transform(train_ys)}
	test_data = {'x' : nonMnist_all['test_dataset'], 'y' : enc.transform(test_ys)}
	return train_data, test_data

def run_nonmnist():
	print 'Running nonMNIST CNN task.'
	print 'Load the nonMNIST data...'
	train_data, test_data = load_all_data('notMNIST_sanitized.pickle')
	return train_data, test_data

if __name__ == '__main__':
	run_nonmnist()