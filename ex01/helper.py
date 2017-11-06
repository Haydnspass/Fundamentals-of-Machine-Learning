from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from scipy.stats import mode
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt

digits = load_digits()

print(digits.keys())

data = digits["data"]
images = digits["images"]
target = digits["target"]
target_names = digits["target_names"]

#print(data.dtype)
#print(data.shape)

img_number = 5
#print(target_names[img_number])
img_shape = [8, 8]

img = images[img_number]  # np.reshape(data[img_number],img_shape)

# test dimensionality
assert 2 == np.size(np.shape(img))

# plt.figure()
# plt.gray()
# plt.imshow(img, interpolation="nearest")
# plt.show()

X_all = data
y_all = target

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(digits.data, digits.target, test_size = 0.4, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.4, random_state=0)

#print(X_train.shape)


# that's an incredible loopy function
def dist_loop(training_set, test_set):
    # train_shape = training
    dist = np.zeros((training_set.shape[0], test_set.shape[0]))
    for i in range(training_set.shape[0]):
        for j in range(test_set.shape[0]):
            dist[i, j] = np.sqrt(np.sum(np.square(test_set[j,k]-training_set[i,k]) for k in range(training_set.shape[1])))
    return dist

def dist_mat(training_set, test_set):
    dist = distance.cdist(training_set, test_set, metric="euclidean")
    return dist

# t = time.process_time()
# dist1 = dist_loop(X_train, X_test)
# elapsed_time = time.process_time() - t
# print("Elapsed time for loopy function was " + str(elapsed_time))

t = time.process_time()
dist2 = dist_mat(X_train, X_test)
elapsed_time = time.process_time() - t
print("Elapsed time for matrix function was " + str(elapsed_time))

# check equality of both functions
# eps = 1e-5
# abs_diff = np.abs(dist1 - dist2)
# assert abs_diff.all() < eps

# implementation of nearest neighbour classifier
dist = dist2

def nearest_neighbour(xt, x, yt): # x: new data, xt: trained data yt: trained target
    dist = dist_mat(xt, x) # rows are xt, columns x, so find argmin in one column
    argmin_dist = np.argmin(dist, 0)
    y = yt[argmin_dist]
    return y

def k_nearest_neighbour(xt, x, yt, k):
    dist = dist_mat(xt, x)  # rows are xt, columns x, so find argmin in one column
    argmin_dist = np.argsort(dist, 0)
    y_argmin_dist = yt[argmin_dist[:k,:]] #[argmin_dist,k]
    # find most common value and output that as majority vote /// tbd
    majority_information = mode(y_argmin_dist, axis=0)
    return np.squeeze(majority_information.mode)
    #return y


# only use 3 and 9
n1 = 3
n2 = 9
sub_ix = np.where((y_train == n1) | (y_train == n2))
y_train_sub = y_train[sub_ix]
X_train_sub = (X_train[sub_ix,:]).squeeze()

sub_ix = np.where((y_test == n1) | (y_test == n2))
y_test_sub = y_test[sub_ix]
X_test_sub = (X_test[sub_ix,:]).squeeze()

def calc_true_pred(y_p, y_t):
    is_true = np.zeros_like(y_t)
    is_true[(y_p == y_t)] = 1
    no_true = np.sum(is_true)
    perc_true = no_true / len(y_p)
    return no_true, perc_true

y_pNN = nearest_neighbour(X_train_sub, X_test_sub, y_train_sub)
y_pkNN = k_nearest_neighbour(X_train_sub, X_test_sub, y_train_sub, 10)

_, percent_true_1NN = calc_true_pred(y_pNN, y_test_sub)
_, percent_true_kNN = calc_true_pred(y_pkNN, y_test_sub)

#print(percent_true_1NN)
#print(percent_true_kNN)


def cross_validation(x,y,k,folds):
    len_data = len(y)
    fold_size = np.floor(len_data / folds)

    percent_true_kNN = np.zeros((folds,1))
    for i in range(folds):
        ix_test = list(range(int(i * fold_size), int((i + 1) * fold_size)))
        ix_train = list(range(len_data))
        del ix_train[int(i * fold_size):int((i + 1) * fold_size)]

        xs_train = x[ix_train]
        xs_test = x[ix_test]

        ys_train = y[ix_train]
        ys_test = y[ix_test]


        result_classifier = k_nearest_neighbour(xs_train,xs_test,ys_train, k)
        _, percent_true_kNN[i] = calc_true_pred(result_classifier, ys_test)
        # print(percent_true_kNN[i])
    mean_perc = np.mean(percent_true_kNN)
    print(mean_perc)

cross_validation(digits.data, digits.target, 5, 10)


#do some stuff


