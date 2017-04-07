from sklearn.linear_model import LogisticRegression
import tensorflow as tf

from data.constants import image_size, num_labels, num_channels
from .helpers import compute_accuracy_multi, accuracy


class LogisticRegressionClassifier:

    def __init__(self):
        self.classifier = LogisticRegression(multi_class='multinomial',
                                             solver='sag', max_iter=300, n_jobs=4)
        self.n = 0
        self.m = 0

    def trains_with(self, X, y):
        """
        Trains the classifier with train data
        :param X: M inputs of N features 
        :param y: M labels 
        """
        self.classifier.fit(X, y)

    def validates_with(self, X_set, y_set):
        predicted_y = self.classifier.predict(X_set)
        for i in range(20):
            print(predicted_y[i], y_set[i])

        total = len(y_set)
        correct_guesses = len(filter(lambda x: x[0] == x[1], zip(y_set, predicted_y)))
        wrong_guesses = total - correct_guesses

        print("Correct Guesses: ", correct_guesses)
        print("Wrong Guesses: ", wrong_guesses)
        print("Accuracy: ", compute_accuracy_multi(correct_guesses, total))
        print()

    def tests_with(self, X_test, y_test):
        print(self.classifier.score(X_test, y_test))


class LogisticRegressionTensorFlowClassifier:

    def __init__(self, alpha=0.5, num_steps=400, l=0.1):
        self.train_subset = 10000
        self.alpha = alpha
        self.num_steps = num_steps
        self._lambda = l

    def run(self, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
        graph = tf.Graph()
        with graph.as_default():
            # Input data.
            # Load the training, validation and test data into constants that are
            # attached to the graph.
            tf_train_dataset = tf.constant(train_dataset[:self.train_subset, :])
            tf_train_labels = tf.constant(train_labels[:self.train_subset])
            tf_valid_dataset = tf.constant(valid_dataset)
            tf_test_dataset = tf.constant(test_dataset)

            # Variables.
            # These are the parameters that we are going to be training. The weight
            # matrix will be initialized using random values following a (truncated)
            # normal distribution. The biases get initialized to zero.
            weights = tf.Variable(
                tf.truncated_normal([image_size * image_size, num_labels]))
            biases = tf.Variable(tf.zeros([num_labels]))

            # Training computation.
            # We multiply the inputs with the weight matrix, and add biases. We compute
            # the softmax and cross-entropy (it's one operation in TensorFlow, because
            # it's very common, and it can be optimized). We take the average of this
            # cross-entropy across all training examples: that's our loss.
            logits = tf.matmul(tf_train_dataset, weights) + biases
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) \
                   + self._lambda * tf.nn.l2_loss(weights) + self._lambda * tf.nn.l2_loss(biases)

            # Optimizer.
            # We are going to find the minimum of this loss using gradient descent.
            optimizer = tf.train.GradientDescentOptimizer(self.alpha).minimize(loss)

            # Predictions for the training, validation, and test data.
            # These are not part of training, but merely here so that we can report
            # accuracy figures as we train.
            train_prediction = tf.nn.softmax(logits)

            logits_valid = tf.matmul(tf_valid_dataset, weights) + biases
            valid_prediction = tf.nn.softmax(logits_valid)
            loss_valid = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=valid_labels, logits=logits_valid))

            test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

            with tf.Session(graph=graph) as session:
                # This is a one-time operation which ensures the parameters get initialized as
                # we described in the graph: random weights for the matrix, zeros for the
                # biases.
                tf.initialize_all_variables().run()
                print('Initialized')
                for step in range(self.num_steps):
                    # Run the computations. We tell .run() that we want to run the optimizer,
                    # and get the loss value and the training predictions returned as numpy
                    # arrays.
                    _, l, predictions = session.run([optimizer, loss, train_prediction])
                    if (step % 10 == 0):
                        print('Loss at step %d: %f' % (step, l))
                        print('Loss at step %d: %f' % (step, loss_valid.eval()))
                        print('Training accuracy: %.1f%%' % accuracy(
                            predictions, train_labels[:self.train_subset, :]))
                        # Calling .eval() on valid_prediction is basically like calling run(), but
                        # just to get that one numpy array. Note that it recomputes all its graph
                        # dependencies.
                        print('Validation accuracy: %.1f%%' % accuracy(
                            valid_prediction.eval(), valid_labels))
                        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
                        print('')


class StochasticGradientTensorFlowClassifier:

    def __init__(self, alpha=0.2, num_steps=3001, batch_size=128, l=0.1):
        self.train_subset = 15000
        self.alpha = alpha
        self.num_steps = num_steps
        self.batch_size = batch_size
        self._lambda = l

    def run(self, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
        graph = tf.Graph()
        print(train_labels[0:100])
        print(train_labels.shape)
        print(num_labels)
        with graph.as_default():
            # Dataset Placeholders
            tf_train_dataset = tf.placeholder(tf.float32, shape=(self.batch_size, image_size * image_size))
            tf_valid_dataset = tf.placeholder(tf.float32, shape=(None, image_size * image_size))
            tf_test_dataset = tf.placeholder(tf.float32, shape=(None, image_size * image_size))

            # Labels Placeholders:
            tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, num_labels))
            tf_valid_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
            tf_test_labels = tf.placeholder(tf.float32, shape=(None, num_labels))

            # Variables.
            weights = tf.Variable(
                tf.truncated_normal([image_size * image_size, num_labels]))
            biases = tf.Variable(tf.zeros([num_labels]))

            # Training computation.
            logits = tf.matmul(tf_train_dataset, weights) + biases
            loss = (tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)
                + self._lambda * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases))))

            # Optimizer.
            optimizer = tf.train.GradientDescentOptimizer(self.alpha).minimize(loss)

            # Predictions for the training, validation, and test data.
            train_prediction = tf.nn.softmax(logits)

            valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
            valid_correct_prediction = tf.equal(tf.argmax(valid_prediction, 1), tf.argmax(tf_valid_labels, 1))
            valid_accuracy = tf.reduce_mean(tf.cast(valid_correct_prediction, "float"))

            test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
            test_correct_prediction = tf.equal(tf.argmax(test_prediction, 1), tf.argmax(tf_test_labels, 1))
            test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, "float"))

            with tf.Session(graph=graph) as session:
                # This is a one-time operation which ensures the parameters get initialized as
                # we described in the graph: random weights for the matrix, zeros for the
                # biases.
                tf.initialize_all_variables().run()
                print('Initialized')
                for step in range(self.num_steps):
                    offset = (step * self.batch_size) % (train_labels.shape[0] - self.batch_size)

                    batch_data = train_dataset[offset:(offset + self.batch_size), :]
                    batch_labels = train_labels[offset:(offset + self.batch_size), :]

                    feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

                    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

                    if (step % 100 == 0):
                        print("Minibatch loss at step %d: %f" % (step, l))
                        print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                        print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval({
                            tf_valid_dataset: valid_dataset
                        }), valid_labels))

                        print("Test accuracy: %.1f%%" % accuracy(valid_prediction.eval({
                            tf_valid_dataset: test_dataset}), test_labels))
                        print('')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class NeuralNetworkClassifier:

    def __init__(self):
        self.learning_rate = 0.001
        self.training_epochs = 3001
        self.batch_size = 128
        self.display_step = 25
        self._lambda = 0.5

    def run(self, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
        n_hidden_1 = 1024
        n_hidden_2 = 300
        n_hidden_3 = 50
        n_input = image_size * image_size
        n_output = num_labels

        tf_train_dataset = tf.placeholder("float", [None, n_input])
        tf_train_labels = tf.placeholder("float", [None, n_output])
        keep_prob = tf.placeholder(tf.float32)
        global_step = tf.Variable(0)

        # Create model
        def multilayer_perceptron(x, weights, biases, l=0):
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            layer_1 = tf.nn.dropout(layer_1, keep_prob)

            # layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            # layer_2 = tf.nn.relu(layer_2)
            #
            # layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
            # layer_3 = tf.nn.relu(layer_3)

            out_layer = tf.matmul(layer_1, weights['out']) + biases['out']

            # Adding regularization costs here:
            out_layer += l * (tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(biases['b1']))
            # out_layer += l * (tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(biases['b2']))
            # out_layer += l * (tf.nn.l2_loss(weights['h3']) + tf.nn.l2_loss(biases['b3']))
            out_layer += l * (tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(biases['out']))

            return out_layer

        weights = {
            'h1': tf.Variable(tf.random_normal([image_size * image_size, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
            'out': tf.Variable(tf.random_normal([n_hidden_1, num_labels])),
        }

        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([n_hidden_3])),
            'out': tf.Variable(tf.random_normal([num_labels])),
        }

        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 100000, 0.96, staircase=True)
        pred = multilayer_perceptron(tf_train_dataset, weights, biases, l=self._lambda)
        predictions_no_lambda = multilayer_perceptron(tf_train_dataset, weights, biases)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=pred))
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)
        # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)  # global_step=global_step)

        train_prediction = tf.nn.softmax(pred)
        correct_prediction = tf.equal(tf.argmax(predictions_no_lambda, 1), tf.argmax(tf_train_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            print('Initialized')

            m = train_dataset.shape[0]

            for epoch in range(self.training_epochs):
                avg_cost = 0.
                total_batch = int(m/self.batch_size)
                for b in xrange(total_batch):
                    offset = b * self.batch_size
                    batch_data = train_dataset[offset:(offset + self.batch_size), :]
                    batch_labels = train_labels[offset:(offset + self.batch_size), :]

                    feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 1}

                    _, l, predictions = sess.run([optimizer, cost, train_prediction], feed_dict=feed_dict)

                    avg_cost += l / total_batch

                if epoch % self.display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
                    print ("Validation Accuracy:", accuracy.eval({tf_train_dataset: valid_dataset,
                                                       tf_train_labels: valid_labels,
                                                       keep_prob: 1.0}))
                    print ("Test Accuracy:", accuracy.eval({tf_train_dataset: test_dataset,
                                                        tf_train_labels: test_labels,
                                                        keep_prob: 1.0}))
                    print()

            print("END: ")
            print ("Accuracy:", accuracy.eval({tf_train_dataset: test_dataset,
                                               tf_train_labels: test_labels,
                                               keep_prob: 1.0}))


class ConvNeuralNetworkClassifier:

    def __init__(self):
        self.learning_rate = 0.001
        self.training_epochs = 101
        self.batch_size = 128
        self.display_step = 25
        self._lambda = 0.5

    def run(self, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
        # batch_size = 16
        # depth = 16
        # num_hidden = 64

        patch_size = 5
        batch_size = 200
        depth1 = 8
        depth2 = 16
        depth3 = 32
        num_hidden_1 = 265

        n_output = num_labels
        graph = tf.Graph()

        def maxpool2d(x, k=2):
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                                  padding='SAME')

        def conv2d(x, weights, biases, s=1):
            conv = tf.nn.conv2d(x, weights, [1, s, s, 1], padding='SAME')
            hidden = conv + biases
            return tf.nn.relu(hidden)

        with graph.as_default():

            tf_train_dataset = tf.placeholder(
                tf.float32, shape=(batch_size, image_size, image_size, num_channels))
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
            tf_valid_dataset = tf.constant(valid_dataset)
            tf_test_dataset = tf.constant(test_dataset)

            layer1_weights = tf.Variable(tf.truncated_normal(
                [patch_size, patch_size, num_channels, depth1], stddev=0.1))
            layer1_biases = tf.Variable(tf.zeros([depth1]))

            layer2_weights = tf.Variable(tf.truncated_normal(
                [patch_size, patch_size, depth1, depth2], stddev=0.1))
            layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth2]))

            layer3_weights = tf.Variable(tf.truncated_normal(
                [patch_size, patch_size, depth2, depth3], stddev=0.1))
            layer3_biases = tf.Variable(tf.constant(1.0, shape=[depth3]))

            layer4_weights = tf.Variable(tf.truncated_normal(
                [512, num_hidden_1], stddev=0.1))  # 4 * 4 * 16, 64
                # [784, num_hidden_1], stddev=0.1))  # 7 * 7 * 16, 64
            layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden_1]))

            layer5_weights = tf.Variable(tf.truncated_normal(
                [num_hidden_1, num_labels], stddev=0.1))  # 64 * 10
            layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

            keep_prob = tf.placeholder(tf.float32)
            global_step = tf.Variable(0)

            def model(data, keep_prob=1.0):
                # Variables.
                hidden = conv2d(data, layer1_weights, layer1_biases, s=1)
                hidden = maxpool2d(hidden)
                hidden = tf.nn.dropout(hidden, keep_prob)

                hidden = conv2d(hidden, layer2_weights, layer2_biases, s=1)
                hidden = maxpool2d(hidden)
                hidden = tf.nn.dropout(hidden, keep_prob)

                hidden = conv2d(hidden, layer3_weights, layer3_biases, s=1)
                hidden = maxpool2d(hidden)
                hidden = tf.nn.dropout(hidden, keep_prob)

                shape = hidden.get_shape().as_list()
                print(shape)
                reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

                hidden = tf.nn.relu(tf.matmul(reshape, layer4_weights) + layer4_biases)
                hidden = tf.nn.dropout(hidden, keep_prob)

                output = tf.matmul(hidden, layer5_weights) + layer5_biases

                return output

            # Training computation.
            logits = model(tf_train_dataset, keep_prob=0.9)
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

            # Learning Rate
            learning_rate = tf.train.exponential_decay(0.03, global_step, 3000, 0.9, staircase=True)

            # Optimizer.
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

            # Predictions for the training, validation, and test data.
            train_prediction = tf.nn.softmax(logits)
            valid_prediction = tf.nn.softmax(model(tf_valid_dataset, keep_prob=1.0))
            test_prediction = tf.nn.softmax(model(tf_test_dataset, keep_prob=1.0))

            num_steps = 40001

            init = tf.initialize_all_variables()
            print('')

            with tf.Session(graph=graph) as session:
                session.run(init)

                for step in range(num_steps):
                    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]

                    batch_labels = train_labels[offset:(offset + batch_size), :]
                    feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

                    _, l, predictions = session.run(
                        [optimizer, loss, train_prediction], feed_dict=feed_dict)

                    if (step % 100 == 0):
                        print('Minibatch loss at step %d: %f' % (step, l))
                        print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                        print('Validation accuracy: %.1f%%' % accuracy(
                            valid_prediction.eval(), valid_labels))
                        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
                        print('Learning rate: ', learning_rate.eval())
                        print('')

                # return layer1_weights.eval(), layer2_weights.eval(), layer3_weights.eval()
