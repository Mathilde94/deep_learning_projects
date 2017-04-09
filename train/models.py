from sklearn.linear_model import LogisticRegression
import tensorflow as tf

from data.constants import image_size, num_labels
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
