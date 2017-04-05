import numpy as np


def compute_precision_and_recall(tp, fp, fn):
    return [tp / (tp+fp), tp / (tp + fn)]


def compute_accuracy(tp, fp, tn, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def compute_accuracy_multi(correct_guesses, total):
    return float(correct_guesses)/ float(total)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])
