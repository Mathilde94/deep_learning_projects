import abc
import tensorflow as tf

from data.constants import  num_labels, num_channels
from model_trainer.constants import classic_batch_size

from .configurations import NeuralNetworkConfiguration, ConvolutionalNeuralNetworkConfiguration
from .helpers import conv2d, maxpool2d
from .parameters import LayerParameter, ModelParameters


class MLModel:
    __metaclass__ = abc.ABCMeta

    def __init__(self, configuration=None):
        self._set_configuration(configuration)
        self.parameters = ModelParameters()

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
    def name(self):
        return self.configuration.get('name', self.__class__.name)

    @property
    def optimizer(self):
        return tf.train.GradientDescentOptimizer


class ConvolutionNeuralNetwork(MLModel):

    hyper_parameters = {
        'batch_size': 100,
        'epochs': 1001,
        'display_epochs': 50,
        'keep_prob': 0.9,
        'learning_rate': 0.1,
        'lambda_rate': 0.0,
        'all_batches': False
    }

    def _set_configuration(self, configuration):
        self.configuration = configuration or ConvolutionalNeuralNetworkConfiguration

    def populate_train_dataset_variables(self):
        image_size = self.configuration['image_size']
        self.parameters.tf_train_dataset = tf.placeholder(
                tf.float32, shape=(self.batch_size, image_size, image_size, num_channels))
        self.parameters.tf_train_labels = tf.placeholder(tf.float32,
                                                         shape=(self.batch_size, num_labels))

    def populate_model_variables(self, from_disk=False):

        for layer in self.configuration['convolutional_layers']:
            l = LayerParameter()
            patch_size = layer['patch_size']
            input_depth = layer['input_depth']
            output_depth = layer['output_depth']
            biases_init_value = layer.get('biases_init_value', 0.0)
            l.weights = tf.Variable(tf.truncated_normal(
                [patch_size, patch_size, input_depth, output_depth], stddev=0.1))
            l.biases = tf.Variable(tf.constant(biases_init_value, shape=[output_depth]))

            self.parameters.convolutional_layers.append(l)

        for layer in self.configuration['fully_connected_layers']:
            l = LayerParameter()
            input_depth = layer['input_depth']
            output_depth = layer['output_depth']

            l.weights = tf.Variable(tf.truncated_normal(
                [input_depth, output_depth], stddev=0.1))
            l.biases = tf.Variable(tf.constant(1.0, shape=[output_depth]))

            self.parameters.fully_connected_layers.append(l)

        # For the final Layer:
        l = LayerParameter()
        input_depth = self.configuration['final_layer']['input_depth']
        output_depth = self.configuration['final_layer']['output_depth']
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


class NeuralNetwork(MLModel):

    hyper_parameters = {
        'batch_size': 128,
        'epochs': 501,  #10001,
        'display_epochs': 50,
        'keep_prob': 0.9,
        'learning_rate': 0.5,
        'lambda_rate': 0.03,
        'all_batches': True
    }

    def _set_configuration(self, configuration):
        self.configuration = configuration or NeuralNetworkConfiguration

    def populate_train_dataset_variables(self):
        image_size = self.configuration['image_size']
        self.parameters.tf_train_dataset = tf.placeholder(
                tf.float32, shape=(None, image_size * image_size))
        self.parameters.tf_train_labels = tf.placeholder(tf.float32,
                                                         shape=(None, num_labels))

    def populate_model_variables(self, from_disk=False):

        for layer in self.configuration['hidden_layers']:
            l = LayerParameter()

            l.weights = tf.Variable(tf.random_normal([layer['input'], layer['output']]))
            l.biases = tf.Variable(tf.random_normal([layer['output']]))
            self.parameters.hidden_layers.append(l)

        # For the final Layer:
        l = LayerParameter()
        input = self.configuration['final_layer']['input']
        output = self.configuration['final_layer']['output']
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
    #
    # @property
    # def optimizer(self):
    #     return tf.train.AdamOptimizer

