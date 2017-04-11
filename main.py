import os
import sys
import time

from data.constants import saved_sessions_root
from data.generate import build
from data.load import (get_testing_set, get_training_set, get_validation_set,
                       load_sets_from_file)
from models.models import ConvolutionNeuralNetwork, LogisticRegression, NeuralNetwork
from classifier.helpers import show_stats_from_file
from classifier.models import Classifier
from model_trainer.models import DataForTrainer, DataSet


def shuffle_data(seed=133):
    from data.generate import (randomize, save_sets_to_file, valid_dataset,
                               valid_labels, train_dataset, train_labels,
                               test_dataset, test_labels)
    import numpy as np

    np.random.seed(seed)

    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
    save_sets_to_file(train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels)


def train_logistic(data_for_trainer):
    model = LogisticRegression()
    classifier = Classifier(model)
    classifier.train(data_for_trainer)
    classifier.stats()


def train_nn(data_for_trainer, is_conv=False):
    model = NeuralNetwork() if not is_conv else ConvolutionNeuralNetwork()
    classifier = Classifier(model)
    classifier.train(data_for_trainer, all_batches=not is_conv)
    classifier.stats()


def show_stats(name='convolution_3_layer.session-stats.pickle'):
    pickle_file = os.path.join(saved_sessions_root, name)
    show_stats_from_file(pickle_file)

if __name__ == '__main__':
    arguments = sys.argv
    if len(sys.argv) < 2:
        sys.exit('Please specify: python main.py [build|reload|train] [--seed] <seed>')

    if arguments[1] == 'build':
        build()
        sys.exit(1)
    elif arguments[1] == 'reload':
        try:
            seed = int(arguments[-1])
            shuffle_data(seed=seed)
        except ValueError, IndexError:
            shuffle_data()
        sys.exit(1)

    # Get the data for the training
    datasets = load_sets_from_file()
    train_set = DataSet(*get_training_set(datasets, size=15000))
    valid_set = DataSet(*get_validation_set(datasets, size=3000))
    test_set = DataSet(*get_testing_set(datasets, size=3000))
    del datasets

    start_time = time.time()

    if arguments[1] == 'train_logistic':
        data_for_trainer = DataForTrainer(train_set, valid_set, test_set)
        train_logistic(data_for_trainer)

    elif arguments[1] == 'train_nn_conv':
        data_for_trainer = DataForTrainer(train_set, valid_set, test_set, with_depth=True)
        train_nn(data_for_trainer, is_conv=True)

    elif arguments[1] == 'train_nn':
        data_for_trainer = DataForTrainer(train_set, valid_set, test_set)
        train_nn(data_for_trainer)

    elif arguments[1] == 'show_stats':
        show_stats()

    print('Took: {}s.'.format(str(time.time() - start_time)))