import os
import sys
import time

from data.constants import saved_sessions_root
from data.generate import build
from data.load import (get_testing_set, get_training_set, get_validation_set,
                       load_sets_from_file, reformat, reformat_3d)
from models.models import ConvolutionNeuralNetwork, NeuralNetwork
from classifier.helpers import show_stats_from_file
from classifier.models import Classifier


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


def train_nn(is_conv=False):
    datasets = load_sets_from_file()

    train_data, train_labels = get_training_set(datasets, size=15000)
    valid_data, valid_labels = get_validation_set(datasets, size=3000)
    test_data, test_labels = get_testing_set(datasets, size=3000)
    del datasets

    if not is_conv:
        train_dataset, train_labels = reformat(train_data, train_labels)
        valid_dataset, valid_labels = reformat(valid_data, valid_labels)
        test_dataset, test_labels = reformat(test_data, test_labels)
    else:
        train_dataset, train_labels = reformat_3d(train_data, train_labels)
        valid_dataset, valid_labels = reformat_3d(valid_data, valid_labels)
        test_dataset, test_labels = reformat_3d(test_data, test_labels)

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    start_time = time.time()
    model = NeuralNetwork() if not is_conv else ConvolutionNeuralNetwork()
    classifier = Classifier(model)
    classifier.train(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset,
                     test_labels, save=False, from_disk=False, all_batches=not is_conv)
    print("Took (in seconds):", time.time() - start_time)
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
    if arguments[1] == 'train_conv_nn':
        train_nn(is_conv=True)
    if arguments[1] == 'train_nn':
        train_nn()
    if arguments[1] == 'show_stats':
        show_stats()
    elif arguments[1] == 'reload':
        try:
            seed = int(arguments[-1])
            shuffle_data(seed=seed)
        except ValueError, IndexError:
            shuffle_data()
