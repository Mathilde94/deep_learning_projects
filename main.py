import sys

from data.generate import build
from data.load import get_testing_set, get_training_set, get_validation_set, load_sets_from_file, reformat, reformat_3d
from train.models import (ConvNeuralNetworkClassifier,
                          LogisticRegressionClassifier, LogisticRegressionTensorFlowClassifier,
                          StochasticGradientTensorFlowClassifier, NeuralNetworkClassifier)


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
    print(train_dataset[0][0][0:10])


def train():
    datasets = load_sets_from_file()

    classifier = LogisticRegressionClassifier()
    valid_data, valid_labels = get_validation_set(datasets, size=10000)
    test_data, test_labels = get_testing_set(datasets, size=10000)

    sizes = [200000]
    for size in sizes:
        print("Computing with training size data: ", size)
        train_data, train_labels = get_training_set(datasets, size=size)
        classifier.trains_with(train_data, train_labels)
        classifier.validates_with(valid_data, valid_labels)
        classifier.tests_with(test_data, test_labels)


def train_with_tensorflow(is_stochastic=False):
    datasets = load_sets_from_file()

    train_data, train_labels = get_training_set(datasets, size=15000)
    valid_data, valid_labels = get_validation_set(datasets, size=3000)
    test_data, test_labels = get_testing_set(datasets, size=3000)
    del datasets

    train_dataset, train_labels = reformat(train_data, train_labels)
    valid_dataset, valid_labels = reformat(valid_data, valid_labels)
    test_dataset, test_labels = reformat(test_data, test_labels)

    print(len(train_dataset))
    print(len(valid_dataset))

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    classifier = StochasticGradientTensorFlowClassifier() if is_stochastic else LogisticRegressionTensorFlowClassifier()
    classifier.run(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)


def train_nn():
    datasets = load_sets_from_file()

    train_data, train_labels = get_training_set(datasets, size=180000)
    valid_data, valid_labels = get_validation_set(datasets, size=20000)
    test_data, test_labels = get_testing_set(datasets, size=20000)
    del datasets

    train_dataset, train_labels = reformat(train_data, train_labels)
    valid_dataset, valid_labels = reformat(valid_data, valid_labels)
    test_dataset, test_labels = reformat(test_data, test_labels)

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    classifier = NeuralNetworkClassifier()
    classifier.run(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)


def train_conv_nn():
    datasets = load_sets_from_file()

    train_data, train_labels = get_training_set(datasets, size=60000)
    valid_data, valid_labels = get_validation_set(datasets, size=12000)
    test_data, test_labels = get_testing_set(datasets, size=12000)
    del datasets

    train_dataset, train_labels = reformat_3d(train_data, train_labels)
    valid_dataset, valid_labels = reformat_3d(valid_data, valid_labels)
    test_dataset, test_labels = reformat_3d(test_data, test_labels)

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    classifier = ConvNeuralNetworkClassifier()
    return classifier.run(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)


if __name__ == '__main__':
    arguments = sys.argv
    if len(sys.argv) < 2:
        sys.exit('Please specify: python main.py [build|reload|train] [--seed] <seed>')
    if arguments[1] == 'build':
        build()
    if arguments[1] == 'train':
        train()
    if arguments[1] == 'train_nn':
        train_nn()
    if arguments[1] == 'train_conv_nn':
        train_conv_nn()
    if arguments[1] == 'train_with_tensorflow':
        train_with_tensorflow(is_stochastic=True)
    elif arguments[1] == 'reload':
        try:
            seed = int(arguments[-1])
            shuffle_data(seed=seed)
        except ValueError, IndexError:
            shuffle_data()
