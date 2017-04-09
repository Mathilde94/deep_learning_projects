import tensorflow as tf


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def conv2d(x, weights, biases, s=1):
    conv = tf.nn.conv2d(x, weights, [1, s, s, 1], padding='SAME')
    hidden = conv + biases
    return tf.nn.relu(hidden)