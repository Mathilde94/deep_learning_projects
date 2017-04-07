import numpy as np
import matplotlib.pyplot as plt


def compute_precision_and_recall(tp, fp, fn):
    return [tp / (tp+fp), tp / (tp + fn)]


def compute_accuracy(tp, fp, tn, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def compute_accuracy_multi(correct_guesses, total):
    return float(correct_guesses)/ float(total)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def show_filter(weights):
    """
    For filters with input depth of 1:
    """
    fig = plt.figure()
    w, h, l, total_filters = weights.shape
    index = 0
    for i in range(total_filters):
        img = weights[:, :, :, i]
        img = img.reshape(w, h)
        index += 1
        a = fig.add_subplot(total_filters, total_filters, index)
        plt.imshow(img)
