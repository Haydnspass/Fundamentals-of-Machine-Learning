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
    # do a 1D histogram


    def freedman_diaconis(X):
        # iqr = X_3/4 quartile - X_1/4 quartile
        return (2 * iqr) / (N ^ (1/3))

if __name__ == '__main__':

    digits = load_digits()

    print(digits.keys())

    data = digits["data"]
    images = digits["images"]
    target = digits["target"]
    target_names = digits["target_names"]
    #
    # x, y = use_subset([1, 7], data, target)
    # x = reduce_dim(x)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.65, random_state=0)