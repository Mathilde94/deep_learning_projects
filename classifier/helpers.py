import os
import matplotlib.pyplot as plt
from matplotlib import pylab

from six.moves import cPickle as pickle

from data.constants import saved_sessions_root


def show_stats(steps, losses, accuracies):
    # % matplotlib inline
    fig = plt.figure()

    fig.add_subplot(4, 1, 1)
    plt.plot(steps, losses['minibatch_train'], 'r--',
             losses['validation'], 'b--',
             losses['test'], 'g--',
             linewidth=1)

    fig.add_subplot(4, 1, 2)
    plt.plot(steps, accuracies['minibatch_train'], 'b--', linewidth=1)

    fig.add_subplot(4, 1, 3)
    plt.plot(steps, accuracies['validation'], 'y--', linewidth=1)

    fig.add_subplot(4, 1, 4)
    plt.plot(steps, accuracies['test'], 'g--', linewidth=1)


def show_stats_from_file(filename):
    pickle_file = os.path.join(saved_sessions_root, filename)
    try:
        stats_data = pickle.load(file(pickle_file))
        show_stats(stats_data['steps'], stats_data['losses'], stats_data['accuracies'])
    except Exception as e:
        print("Can not show the stats in fileL ", pickle_file)
        raise e


def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(15,15))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i,:]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',ha='right', va='bottom')
    pylab.show()
