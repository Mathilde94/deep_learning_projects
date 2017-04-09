import os
import numpy as np

from six.moves import cPickle as pickle

from .constants import data_root, image_size, num_labels, num_channels


def load_from_file(pickle_file):
    return pickle.load(file(pickle_file))


def load_sets_from_file():
    pickle_file = os.path.join(data_root, 'notMMIST-{}.pickle'.format(num_labels))
    return load_from_file(pickle_file)


def get_sets_of_type(datasets, type='train', size=50):
    dataset = np.array(datasets[type + '_dataset'][0: size])
    labels = np.array(datasets[type + '_labels'][0: size])

    m, n1, n2 = dataset.shape
    dataset_reshaped = np.reshape(dataset, (m, n1 * n2))
    return [dataset_reshaped, labels]


def get_training_set(datasets, size=50):
    return get_sets_of_type(datasets, type='train', size=size)


def get_validation_set(datasets, size=50):
    return get_sets_of_type(datasets, type='valid', size=size)


def get_testing_set(datasets, size=50):
    return get_sets_of_type(datasets, type='test', size=size)


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def reformat_3d(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels
