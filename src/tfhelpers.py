'''
Created on Oct 4, 2016

@author: pjmartin (but mostly the Google TF tutorial site)
'''
import tensorflow as tf

# variable declaration helper functions
# Make the weight variables be a little noisy around 0.0.
def weight_variable(shape, var_name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=var_name)

# creates a small positive bias
def bias_variable(shape, var_name):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=var_name)

# variable_summaries collects the mean, stdev, max, and min
# values of the supplied variable, var.
def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
#         tf.histogram_summary(name, var)

# fully_conn_nn_layer generates a fully connected
# NN layer based on the supplied input_tensor, input, and output dims.
def fully_conn_nn_layer(input_tensor, in_dim, out_dim, layer_name, act_fn=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            W = tf.Variable(tf.truncated_normal([in_dim, out_dim]))
            variable_summaries(W, layer_name + '/weights')
        with tf.name_scope('biases'):
            b = tf.Variable(tf.random_normal([out_dim]))
            variable_summaries(b, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            pre_act = tf.matmul(input_tensor, W) + b
            tf.histogram_summary(layer_name + '/pre_acts_hist', pre_act)
        acts = act_fn(pre_act, 'activations')
        tf.histogram_summary(layer_name + '/activations', acts)
        return acts, W, b

# Support functions for convolutional NN
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# Maxpooling over 2x2 blocks
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
