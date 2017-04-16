import numpy as np
import os
import tensorflow as tf
from six.moves import cPickle as pickle

from data.load import reformat, reformat_with_depth
from data.text_load import generate_batch, valid_size

from .constants import (classic_batch_size, display_epochs, epochs,
                        keep_prob, lambda_rate, learning_rate)
from .helpers import accuracy, BatchGenerator, characters, logprob, random_distribution, sample


class DataSet:

    def __init__(self, input, output):
        self.input = input
        self.output = output

    def format(self, with_depth=False):
        return self.reformat_with_depth() if with_depth else self.reformat()

    def reformat(self):
        return reformat(self.input, self.output)

    def reformat_with_depth(self):
        return reformat_with_depth(self.input, self.output)


class DataForTrainer:

    def __init__(self, train_set, validation_set, test_set, with_depth=False):
        self.train_set = train_set
        self.validation_set = validation_set
        self.test_set = test_set
        self._format_sets(with_depth=with_depth)

    def _format_sets(self, with_depth=False):
        self.train_set = DataSet(*self.train_set.format(with_depth))
        self.validation_set = DataSet(*self.validation_set.format(with_depth))
        self.test_set = DataSet(*self.test_set.format(with_depth))

    def print_shapes(self):
        print('Training set', self.train_set.input.shape, self.train_set.output.shape)
        print('Validation set', self.validation_set.input.shape, self.validation_set.output.shape)
        print('Test set', self.test_set.input.shape, self.test_set.output.shape)
        print('')

    def get_all_sets(self):
        return (self.train_set.input, self.train_set.output, self.validation_set.input,
                self.validation_set.output, self.test_set.input, self.test_set.output)


class BaseTrainer:
    batch_size = classic_batch_size

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

    def set_training_hyper_parameters(self, parameters):
        for param, value in parameters.items():
            setattr(self, param, value)

    def save_session(self, session, filename):
        save_path = self.saver.save(session, filename)
        print("Session saved in file: %s" % save_path)

    def save_stats(self, filename):

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


class Trainer(BaseTrainer):

    display_epochs = display_epochs
    epochs = epochs
    lambda_rate = lambda_rate
    learning_rate = learning_rate
    keep_prob = keep_prob

    def run(self, model, data, from_disk=False, save=False, all_batches=False, session_file=''):
        train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = data.get_all_sets()

        graph = tf.Graph()
        with graph.as_default():
            model.populate_train_dataset_variables()
            model.populate_model_variables()

            tf_train_dataset = model.parameters.tf_train_dataset
            tf_train_labels = model.parameters.tf_train_labels

            tf_valid_dataset = tf.constant(valid_dataset)
            tf_test_dataset = tf.constant(test_dataset)

            global_step = tf.Variable(0)

            logits = model.feed_forward(tf_train_dataset, keep_prob=self.keep_prob, _lambda=self.lambda_rate)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,
                                                                          logits=logits))
            # Learning Rate
            learning_rate = tf.train.exponential_decay(self.learning_rate, global_step,
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
                for step in range(self.epochs):

                    if all_batches:
                        avg_cost = 0.
                        total_batch = int(m/self.batch_size)
                        for b in xrange(total_batch):
                            offset = b * self.batch_size
                            batch_data = train_dataset[offset:(offset + self.batch_size), :]
                            batch_labels = train_labels[offset:(offset + self.batch_size), :]

                            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

                            _, l, predictions = session.run([optimizer, loss, train_prediction],
                                                            feed_dict=feed_dict)

                            avg_cost += l / total_batch
                    else:
                        offset = (step * self.batch_size) % (train_labels.shape[0] - self.batch_size)
                        batch_data = train_dataset[offset:(offset + self.batch_size), :]  #, :, :]
                        batch_labels = train_labels[offset:(offset + self.batch_size), :]

                        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

                        _, l, predictions = session.run([optimizer, loss, train_prediction],
                                                        feed_dict=feed_dict)

                    if step % self.display_epochs == 0:
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
                    self.save_stats(session_file)


class TextTrainer(BaseTrainer):

    def run(self, model, data, from_disk=False, save=False, all_batches=False, session_file=''):

        graph = tf.Graph()
        with graph.as_default():
            model.populate_train_dataset_variables()
            model.populate_model_variables()

            tf_train_dataset = model.parameters.tf_train_dataset
            tf_train_labels = model.parameters.tf_train_labels
            tf_valid_dataset = tf.constant(model.parameters.valid_examples, dtype=tf.int32)

            embed = model.feed_forward(tf_train_dataset)
            loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(weights=model.softmax_weights, biases=model.softmax_biases,
                                           inputs=embed, labels=tf_train_labels, num_sampled=self.num_sampled,
                                           num_classes=model.vocabulary_size))
            optimizer = model.optimizer(self.learning_rate).minimize(loss)

            # Compute the similarity between minibatch examples and all embeddings.
            # We use the cosine distance:
            norm = tf.sqrt(tf.reduce_sum(tf.square(model.embeddings), 1, keep_dims=True))
            normalized_embeddings = model.embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, tf_valid_dataset)
            similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

            if not from_disk:
                init = tf.initialize_all_variables()

            self.saver = tf.train.Saver()

            with tf.Session(graph=graph) as session:
                if not from_disk:
                    session.run(init)
                else:
                    self.saver.restore(session, session_file)

                average_loss = 0
                for step in range(self.epochs):
                    batch_data, batch_labels = generate_batch(data, self.batch_size, self.num_skips, self.skip_window)
                    feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
                    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
                    average_loss += l
                    if step % 2000 == 0:
                        if step > 0:
                            average_loss = average_loss / 2000
                        # The average loss is an estimate of the loss over the last 2000 batches.
                        print('Average loss at step %d: %f' % (step, average_loss))
                        self.steps.append(step)
                        self.losses['minibatch_train'].append(average_loss)

                        average_loss = 0
                    # note that this is expensive (~20% slowdown if computed every 500 steps)
                    if step % 10000 == 0:
                        sim = similarity.eval()
                        for i in range(model.valid_size):
                            valid_word = self.reverse_dictionary[model.parameters.valid_examples[i]]
                            top_k = 8  # number of nearest neighbors
                            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                            log = 'Nearest to %s:' % valid_word
                            for k in range(top_k):
                                close_word = self.reverse_dictionary[nearest[k]]
                                log = '%s %s,' % (log, close_word)
                            print(log)

                if save:
                    self.save_session(session, session_file)
                    self.save_stats(session_file)

                return normalized_embeddings.eval()


class LSTMTrainer(BaseTrainer):

    def run(self, model, train_set, valid_set, all_batches=False, from_disk=False, save=False, session_file=''):

        graph = tf.Graph()
        with graph.as_default():
            model.populate_train_dataset_variables()
            model.populate_model_variables()

            train_batches = BatchGenerator(train_set, model.batch_size, model.num_unrollings)
            valid_batches = BatchGenerator(valid_set, 1, 1)

            # Unrolled LSTM loop.
            outputs = list()
            output = model.parameters.saved_output
            state = model.parameters.saved_state
            for i in model.parameters.train_inputs:
                output, state = model.feed_forward(i, output, state)
                outputs.append(output)

            # State saving across unrollings.
            with tf.control_dependencies([model.parameters.saved_output.assign(output),
                                          model.parameters.saved_state.assign(state)]):
                # Classifier.
                o = tf.reshape(tf.pack(outputs, 0), [model.num_unrollings * model.batch_size, model.num_nodes])
                logits = tf.nn.xw_plus_b(o, model.parameters.w, model.parameters.b)
                l = tf.reshape(tf.pack(model.parameters.train_labels, 0),
                               [model.num_unrollings * model.batch_size, model.vocabulary_size])
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=l, logits=logits))

            # Optimizer.
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 5000, 0.1, staircase=True)
            optimizer = model.optimizer(learning_rate)
            gradients, v = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
            optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

            train_prediction = tf.nn.softmax(logits)

            # Sampling and validation eval: batch 1, no unrolling.
            sample_input = tf.placeholder(tf.float32, shape=[1, model.vocabulary_size])
            saved_sample_output = tf.Variable(tf.zeros([1, model.num_nodes]))
            saved_sample_state = tf.Variable(tf.zeros([1, model.num_nodes]))
            reset_sample_state = tf.group(
                saved_sample_output.assign(tf.zeros([1, model.num_nodes])),
                saved_sample_state.assign(tf.zeros([1, model.num_nodes])))
            sample_output, sample_state = model.feed_forward(sample_input, saved_sample_output, saved_sample_state)
            with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                          saved_sample_state.assign(sample_state)]):
                sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, model.parameters.w, model.parameters.b))

            if not from_disk:
                init = tf.initialize_all_variables()

            self.saver = tf.train.Saver()

            with tf.Session(graph=graph) as session:
                if not from_disk:
                    session.run(init)
                else:
                    self.saver.restore(session, session_file)

                mean_loss = 0
                for step in range(self.epochs):
                    batches = train_batches.next()
                    feed_dict = dict()
                    for i in range(model.num_unrollings + 1):
                        feed_dict[model.parameters.train_data[i]] = batches[i]
                    _, l, predictions, lr = session.run(
                        [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
                    mean_loss += l
                    if step % self.display_epochs == 0:
                        if step > 0:
                            mean_loss = mean_loss / self.display_epochs
                        # The mean loss is an estimate of the loss over the last few batches.
                        print(
                            'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
                        mean_loss = 0
                        labels = np.concatenate(list(batches)[1:])
                        print('Minibatch perplexity: %.2f' % float(
                            np.exp(logprob(predictions, labels))))
                        if step % (self.display_epochs * 10) == 0:
                            # Generate some samples.
                            print('=' * 80)
                            for _ in range(5):
                                feed = sample(random_distribution())
                                sentence = characters(feed)[0]
                                reset_sample_state.run()
                                for _ in range(79):
                                    prediction = sample_prediction.eval({sample_input: feed})
                                    feed = sample(prediction)
                                    sentence += characters(feed)[0]
                                print(sentence)
                            print('=' * 80)
                        # Measure validation set perplexity.
                        reset_sample_state.run()
                        valid_logprob = 0
                        for _ in range(valid_size):
                            b = valid_batches.next()
                            predictions = sample_prediction.eval({sample_input: b[0]})
                            valid_logprob = valid_logprob + logprob(predictions, b[1])
                        print('Validation set perplexity: %.2f' % float(np.exp(
                            valid_logprob / valid_size)))

                if save:
                    self.save_session(session, session_file)
                    self.save_stats(session_file)