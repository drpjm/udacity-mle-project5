'''
Created on Oct 22, 2016

@author: pjmartin
'''
import scipy.ndimage as img
import scipy.misc as misc
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import os
import DigitStructFile
import sklearn.preprocessing as skproc
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
import cPickle as pkl

# train_loc_root = '/Users/pjmartin/Documents/Udacity/MachineLearningProgram/Project5/udacity-mle-project5/data/train/'
# test_loc_root = '/Users/pjmartin/Documents/Udacity/MachineLearningProgram/Project5/udacity-mle-project5/data/test/'

# Crops a square, but may be skewed if the square goes outside of the top or left axis.
def square_dims(img_dict):
    mintop = np.min(img_dict['top'])
    minleft = np.min(img_dict['left'])
    height = np.max(img_dict['height'])
    width = np.sum(img_dict['width'])
    center_from_left = minleft + np.floor(width / 2.0)
    center_from_top = mintop + np.floor(height / 2.0)
    max_dim = max([height, width]) + 0.1*(max([height, width]))
    new_left = int(max([0, center_from_left - np.floor(max_dim/2.0)]))
    new_top = int(max([0, center_from_top - np.floor(max_dim/2.0)]))
    return [new_top, new_left, int(new_top + max_dim), int(new_left + max_dim)]

# This function extracts and crops the indexed image.
def extract_and_crop(img_loc, img_dict, resize):
    curr_img = img.imread(img_loc + img_dict['name'],mode='L')
    img_squaredims = square_dims(img_dict)
    # bb_top = np.min(img_dict['top'])
    # bb_left = np.min(img_dict['left'])
    # bb_height = np.max(img_dict['height'])
    # bb_twidth = np.sum(img_dict['width'])
    # Add some pixel buffer before cropping.
    # min_top = int( bb_top - 0.1*bb_top )
    # min_left = int( bb_left - 0.1*bb_left )
    # if min_left < 0:
    #     min_left = 0
    # if min_top < 0:
    #     min_top = 0
    # ... a little less on the height
    # total_height = int( bb_height + 0.05*bb_height )
    # total_width = int( bb_twidth + 0.15*bb_twidth )

    curr_img_crop = curr_img[img_squaredims[0]:img_squaredims[2], img_squaredims[1]:img_squaredims[3]]
    # curr_img_crop = curr_img[min_top:min_top+total_height,min_left:min_left+total_width]

    img_rs = misc.imresize(curr_img_crop, (resize,resize))
    return img_rs

# This function takes a digit struct and creates a one hot encoding of the data.
def extract_label(img_dict, encoder, max_len):
    # Build the label data with one hot encoding.
    street_label = np.array(img_dict['label']).astype(int)
    # Replace any instances of 10 with 0 - needed for one-hot encoding.
    street_label[street_label == 10] = 0
    curr_len = np.shape(street_label)[0]
    len_onehot = encoder.fit_transform(curr_len)
    y_onehot = np.concatenate((len_onehot, encoder.fit_transform(street_label.reshape(-1,1))),axis=0)
    # Create the padding for MAX_LENGTH - curr_len
    if max_len - curr_len > 0:
        nodigit_padding = np.array([10 for i in range(max_len-curr_len)])
        padding_onehot = encoder.fit_transform(nodigit_padding.reshape(-1,1))
        y_onehot = np.concatenate((y_onehot, padding_onehot), axis=0)
    return y_onehot

def generate_svhn_dataset(file_loc, n_vals, n_labels, crop_size, max_len):
    # Load from the digitstruct mat file.
    fname = os.path.join(file_loc, "digitStruct.mat")
    digitstruct = DigitStructFile(fname)
    data_len = len(digitstruct.digitStructName)
    X = np.zeros((data_len, crop_size, crop_size))
    y = np.zeros((data_len, n_labels, n_vals))

    invalid_idxs = []
    # Encoder for label generation.
    enc = skproc.OneHotEncoder(n_values=n_vals,sparse=False)
    for i in range(data_len):
        curr_dict = digitstruct.getDigitStructure(i)
        street_num_len = len(np.array(curr_dict['label']))
        if i % 1000 == 0:
            print "Processed through image " + str(i)
        if street_num_len <= 5 and street_num_len > 0:
            # Extract the label
            curr_y = extract_label(curr_dict, enc, max_len)
            curr_X = extract_and_crop(file_loc, curr_dict, crop_size)
            y[i,:,:] = curr_y
            X[i,:,:] = curr_X
        else:
            invalid_idxs.append(i)
            print "Invalid number! Index = " + str(i)
    X = np.delete(X,invalid_idxs,axis=0)
    y = np.delete(y,invalid_idxs,axis=0)
    return { 'data' : X, 'labels' : y }

def pickle_svhn(name, dataset):
    fname = "svhn_" + name + ".pkl"
    svhn_pkl_file = open(fname, 'wb')
    pkl.dump(dataset, svhn_pkl_file, -1)
    svhn_pkl_file.close()

def load_svhn_pkl(fname):
    svhn_pkl_file = open(fname, 'rb')
    loaded_dataset = pkl.load(svhn_pkl_file)
    svhn_pkl_file.close()
    return loaded_dataset

# Invokes the preprocessing algorithms above on training and test data.
def build_data_sets(train_loc_root, test_loc_root):
	test_dataset = generate_svhn_dataset(test_loc_root, 11, 6, 32, 5)
	pickle_svhn("test", test_dataset)
	train_dataset = generate_svhn_dataset(train_loc_root, 11, 6, 32, 5)
	pickle_svhn("train", train_dataset)
