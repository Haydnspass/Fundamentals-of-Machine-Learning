from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from scipy.stats import mode
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt


# condition is or
def use_subset(condition, x, y):
    sub_ix = np.where((y == condition[0]) | (y == condition[1]))
    y_sub = (y[sub_ix]).squeeze()
    x_sub = (x[sub_ix, :]).squeeze()
    return x_sub, y_sub


def fit_naive_bayes(features, labels, bincount=0):
    def filter_data(features, labels, c):
        assert c in labels, 'Class is not in labels'
        sub_ix = c == labels
        return features[sub_ix], labels[sub_ix]

    def histogram(xx, num_bins):
        h, b = np.histogram(xx, bins=num_bins)
        return h, b

    def freedman_diaconis(X):
        iqr = np.percentile(X, 75) - np.percentile(X, 25)  # X_3/4 quartile - X_1/4 quartile
        if iqr == 0:
            delta_x = float('nan')
        else:
            delta_x = (2 * iqr) / (np.power(len(X), 1 / 3))
        return delta_x

    if bincount == 0:
        # loop over features and use reasonable freedman diaconis
        bincount_helper = np.zeros(features.shape[1])
        for i in range(features.shape[1]):
            delta_x = freedman_diaconis(features[:, i])
            bincount_helper[i] = np.ceil((np.max(features[:, i]) - np.min(features[:, i])) / delta_x)
        # use median value neglecting nan
        bincount = int(np.ceil(np.median(bincount_helper[np.invert(np.isnan(bincount_helper))])))

    hist = np.zeros((len(labels), features.shape[1], bincount))
    binning = np.zeros((len(labels), features.shape[1], bincount + 1))
    for i, c in enumerate(labels):
        features_sub, _ = filter_data(features, labels, c)
        hist[i,:,:], binning[i,:,:] = histogram(features_sub, bincount)
    return hist, binning


def predict_naive_bayes(features, hist, binning):
    l = np.floor((features - binning[:,:,0])/binning[:,:,1])
    print('hhh')
    return l


if __name__ == '__main__':

    digits = load_digits()

    print(digits.keys())

    data = digits["data"]
    images = digits["images"]
    target = digits["target"]
    target_names = digits["target_names"]
    #
    x, y = use_subset([1, 7], data, target)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    hist, binning = fit_naive_bayes(X_train, y_train)
    predict_naive_bayes(X_test, hist, binning)