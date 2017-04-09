import os
import tensorflow as tf
from six.moves import cPickle as pickle


from .constants import (classic_batch_size, display_epochs, epochs,
                        keep_prob, lambda_rate, learning_rate)
from .helpers import accuracy


class ModelTrainer:

    def __init__(self):
        self.saver = None

        self._init_stats()

    def _init_stats(self):
        self.accuracies = {
            'minibatch_train': [],
            'validation': [],
            'test': []
        }
        self.losses = {
            'minibatch_train': [],
            'validation': [],
            'test': []
        }
        self.steps = []

    def run(self, model, train_dataset, train_labels, valid_dataset,
            valid_labels, test_dataset, test_labels,
            from_disk=False, save=False, session_file='', all_batches=False, display_epochs=display_epochs, epochs=epochs,
                 lambda_rate=lambda_rate, learning_rate=learning_rate, keep_prob=keep_prob, batch_size=classic_batch_size):
        graph = tf.Graph()

        with graph.as_default():
            model.populate_train_dataset_variables()
            model.populate_model_variables()

            tf_train_dataset = model.parameters.tf_train_dataset
            tf_train_labels = model.parameters.tf_train_labels

            tf_valid_dataset = tf.constant(valid_dataset)
            tf_test_dataset = tf.constant(test_dataset)

            global_step = tf.Variable(0)

            logits = model.feed_forward(tf_train_dataset, keep_prob=keep_prob, _lambda=lambda_rate)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,
                                                                          logits=logits))
            # Learning Rate
            learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                                       10000, 0.96, staircase=True)

            # Optimizer.
            optimizer = model.optimizer(learning_rate).minimize(loss, global_step=global_step)

            # Predictions for the training, validation, and test data.
            train_prediction = tf.nn.softmax(logits)
            valid_prediction = tf.nn.softmax(model.feed_forward(tf_valid_dataset))
            test_prediction = tf.nn.softmax(model.feed_forward(tf_test_dataset))

            if not from_disk:
                init = tf.initialize_all_variables()

            self.saver = tf.train.Saver()

            with tf.Session(graph=graph) as session:
                if not from_disk:
                    session.run(init)
                else:
                    self.saver.restore(session, session_file)

                m = train_dataset.shape[0]
                for step in range(epochs):

                    if all_batches:
                        avg_cost = 0.
                        total_batch = int(m/batch_size)
                        for b in xrange(total_batch):
                            offset = b * batch_size
                            batch_data = train_dataset[offset:(offset + batch_size), :]
                            batch_labels = train_labels[offset:(offset + batch_size), :]

                            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

                            _, l, predictions = session.run([optimizer, loss, train_prediction],
                                                            feed_dict=feed_dict)

                            avg_cost += l / total_batch
                    else:
                        # TODO: so far only for convolutional networks
                        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
                        batch_labels = train_labels[offset:(offset + batch_size), :]

                        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

                        _, l, predictions = session.run([optimizer, loss, train_prediction],
                                                        feed_dict=feed_dict)

                    if step % display_epochs == 0:
                        l_validation = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            labels=valid_labels, logits=model.feed_forward(tf_valid_dataset)))
                        l_test = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            labels=test_labels, logits=model.feed_forward(tf_test_dataset)))

                        minibatch_accuracy = accuracy(predictions, batch_labels)
                        valid_accuracy = accuracy(valid_prediction.eval(), valid_labels)
                        test_accuracy = accuracy(test_prediction.eval(), test_labels)

                        self.steps.append(step)
                        self.losses['minibatch_train'].append(l)
                        self.losses['validation'].append(l_validation.eval())
                        self.losses['test'].append(l_test.eval())
                        self.accuracies['minibatch_train'].append(minibatch_accuracy)
                        self.accuracies['validation'].append(valid_accuracy)
                        self.accuracies['test'].append(test_accuracy)

                        print('Step: %d: l=%f l_valid=%f l_test=%f minibatch=%.1f%% valid=%.1f%% test=%.1f%%'
                              % (step, l, l_validation.eval(),l_test.eval(),  minibatch_accuracy,
                                 valid_accuracy, test_accuracy))

                if save:
                    self.save_session(session, session_file)

    def save_session(self, session, filename):
        save_path = self.saver.save(session, filename)
        print("Session saved in file: %s" % save_path)

        pickle_file = os.path.join('{}-stats.pickle'.format(filename))
        try:
            f = open(pickle_file, 'wb')
            stats_data = {
                'steps': self.steps,
                'losses': self.losses,
                'accuracies': self.accuracies
            }
            pickle.dump(stats_data, f, pickle.HIGHEST_PROTOCOL)
            f.close()
            print("Stats saved in : %s" % pickle_file)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

