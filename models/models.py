import abc
import math
import numpy as np
import tensorflow as tf
import random

from data.constants import num_labels, num_channels
from model_trainer.constants import classic_batch_size

from .configurations import (LogisticRegressionConfiguration, NeuralNetworkConfiguration,
                             ConvolutionalNeuralNetworkConfiguration, Word2VecConfiguration)
from .helpers import conv2d, maxpool2d
from .parameters import LayerParameter, ModelParameters


class MLModel:
    __metaclass__ = abc.ABCMeta

    def __init__(self, configuration=None):
        self._set_configuration(configuration)
        self.parameters = ModelParameters()
        self._set_configuration_parameters()

    @abc.abstractmethod
    def _set_configuration(self, configuration):
        pass

    @abc.abstractmethod
    def feed_forward(self, **kwargs):
        pass

    @abc.abstractmethod
    def populate_train_dataset_variables(self):
        pass

    @abc.abstractmethod
    def populate_model_variables(self):
        pass

    @property
    def batch_size(self):
        return self.configuration.get('batch_size', classic_batch_size)

    @property
    def optimizer(self):
        return tf.train.GradientDescentOptimizer

    def populate_train_dataset_variables(self):
        image_size = self.image_size
        self.parameters.tf_train_dataset = tf.placeholder(
                tf.float32, shape=(self.batch_size, image_size * image_size))
        self.parameters.tf_train_labels = tf.placeholder(tf.float32,
                                                         shape=(self.batch_size, num_labels))

    def _set_configuration_parameters(self):
        for param, value in self.configuration.items():
            setattr(self, param, value)


class LogisticRegression(MLModel):

    hyper_parameters = {
        'epochs': 100,
        'display_epochs': 10,
        'keep_prob': 0.9,
        'learning_rate': 0.5,
        'lambda_rate': 0.1,
        'all_batches': False
    }

    def _set_configuration(self, configuration):
        self.configuration = configuration or LogisticRegressionConfiguration

    def populate_model_variables(self, from_disk=False):
        l = LayerParameter()
        layer = self.final_layer
        l.weights = tf.Variable(
            tf.truncated_normal([layer['input_depth'], layer['output_depth']]))
        l.biases = tf.Variable(tf.zeros([layer['output_depth']]))

        self.parameters.final_layer = l

    def feed_forward(self, data, keep_prob=1.0, _lambda=0.0):
        layer = self.parameters.final_layer
        output = tf.matmul(data, layer.weights) + layer.biases

        output += _lambda * (tf.nn.l2_loss(layer.weights) + tf.nn.l2_loss(layer.biases))
        return output


class NeuralNetwork(MLModel):

    hyper_parameters = {
        'epochs': 301,  # 10001,
        'display_epochs': 50,
        'keep_prob': 0.9,
        'learning_rate': 0.5,
        'lambda_rate': 0.03,
        'all_batches': True
    }

    def _set_configuration(self, configuration):
        self.configuration = configuration or NeuralNetworkConfiguration

    def populate_model_variables(self, from_disk=False):

        for layer in self.hidden_layers:
            l = LayerParameter()

            l.weights = tf.Variable(tf.random_normal([layer['input'], layer['output']]))
            l.biases = tf.Variable(tf.random_normal([layer['output']]))
            self.parameters.hidden_layers.append(l)

        # For the final Layer:
        l = LayerParameter()
        input = self.final_layer['input']
        output = self.final_layer['output']
        l.weights = tf.Variable(tf.random_normal([input, output]))
        l.biases = tf.Variable(tf.random_normal([output]))
        self.parameters.final_layer = l

    def feed_forward(self, data, keep_prob=1.0, _lambda=0.0):
        hidden = data
        for layer in self.parameters.hidden_layers:
            hidden = tf.add(tf.matmul(hidden, layer.weights), layer.biases)
            hidden = tf.nn.relu(hidden)
            hidden = tf.nn.dropout(hidden, keep_prob)

        layer = self.parameters.final_layer
        hidden = tf.add(tf.matmul(hidden, layer.weights), layer.biases)

        # Adding regularization costs here:
        output = hidden
        for layer in self.parameters.hidden_layers + [self.parameters.final_layer]:
            output += _lambda * (tf.nn.l2_loss(layer.weights) + tf.nn.l2_loss(layer.biases))
        return output


class ConvolutionNeuralNetwork(MLModel):

    hyper_parameters = {
        'epochs': 301,  #1001,
        'display_epochs': 50,
        'keep_prob': 0.9,
        'learning_rate': 0.1,
        'lambda_rate': 0.0,
        'all_batches': False
    }

    def _set_configuration(self, configuration):
        self.configuration = configuration or ConvolutionalNeuralNetworkConfiguration

    def populate_train_dataset_variables(self):
        image_size = self.image_size
        self.parameters.tf_train_dataset = tf.placeholder(
                tf.float32, shape=(self.batch_size, image_size, image_size, num_channels))
        self.parameters.tf_train_labels = tf.placeholder(tf.float32,
                                                         shape=(self.batch_size, num_labels))

    def populate_model_variables(self, from_disk=False):

        for layer in self.convolutional_layers:
            l = LayerParameter()
            patch_size = layer['patch_size']
            input_depth = layer['input_depth']
            output_depth = layer['output_depth']
            biases_init_value = layer.get('biases_init_value', 0.0)
            l.weights = tf.Variable(tf.truncated_normal(
                [patch_size, patch_size, input_depth, output_depth], stddev=0.1))
            l.biases = tf.Variable(tf.constant(biases_init_value, shape=[output_depth]))

            self.parameters.convolutional_layers.append(l)

        for layer in self.fully_connected_layers:
            l = LayerParameter()
            input_depth = layer['input_depth']
            output_depth = layer['output_depth']

            l.weights = tf.Variable(tf.truncated_normal(
                [input_depth, output_depth], stddev=0.1))
            l.biases = tf.Variable(tf.constant(1.0, shape=[output_depth]))

            self.parameters.fully_connected_layers.append(l)

        # For the final Layer:
        l = LayerParameter()
        input_depth = self.final_layer['input_depth']
        output_depth = self.final_layer['output_depth']
        l.weights = tf.Variable(tf.truncated_normal(
            [input_depth, output_depth], stddev=0.1))
        l.biases = tf.Variable(tf.constant(1.0, shape=[output_depth]))
        self.parameters.final_layer = l

    def feed_forward(self, data, keep_prob=1.0, _lambda=0.0):
        x = data
        for layer in self.parameters.convolutional_layers:
            hidden = conv2d(x, layer.weights, layer.biases, s=1)
            hidden = maxpool2d(hidden)
            hidden = tf.nn.dropout(hidden, keep_prob)
            x = hidden

        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

        for layer in self.parameters.fully_connected_layers:
            hidden = tf.nn.relu(tf.matmul(reshape, layer.weights) + layer.biases)
            hidden = tf.nn.dropout(hidden, keep_prob)

        output = (tf.matmul(hidden, self.parameters.final_layer.weights)
                  + self.parameters.final_layer.biases)
        return output


class Word2Vec(MLModel):

    hyper_parameters = {
        'epochs': 100001,
        'display_epochs': 10000,
        'learning_rate': 0.1,
        'lambda_rate': 0.0,
        'all_batches': True,
        'num_sampled': 64,
        'skip_window': 1,
        'num_skips': 2,
    }

    def _set_configuration(self, configuration):
        self.configuration = configuration or Word2VecConfiguration

    def populate_train_dataset_variables(self):
        valid_window, valid_size = self.valid_window, self.valid_size
        self.parameters.valid_examples = np.array(random.sample(range(valid_window), valid_size))
        self.parameters.tf_train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.parameters.tf_train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

    def populate_model_variables(self, from_disk=False):
        # Variables.
        self.embeddings = tf.Variable(
            tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
        self.softmax_weights = tf.Variable(
            tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                stddev=1.0 / math.sqrt(self.embedding_size)))
        self.softmax_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

    def feed_forward(self, data, keep_prob=1.0, _lambda=0.0):
        return tf.nn.embedding_lookup(self.embeddings, data)

    @property
    def optimizer(self):
        return tf.train.AdagradOptimizer
