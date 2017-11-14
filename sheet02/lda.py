import numpy as np


def fit_lda(training_features, training_labels):
    # calculate means
    mean = np.zeros_like(training_features)
    # returns sorted array of labels
    label_set = np.unique(training_labels)
    mean_vec = np.zeros((len(label_set),training_features.shape[1]))
    for i, label in enumerate(label_set):
        sub_ix = (training_labels == label)
        sub_set = training_features[sub_ix, :]
        mean_vec[i,:] = np.mean(sub_set, axis=0)
        mean[sub_ix, :] = mean_vec[i,:]

    z = np.subtract(training_features, mean)
    z = np.expand_dims(z, axis=1)
    sigma = np.mean([np.dot(z[i].T, z[i]) for i in range(len(z))], 0)
    return mean_vec, sigma, label_set

A = np.array([[1,100],[2,2],[3,3], [4,4]])
l = np.array([1,1,2,1])

mu, sig, labels = fit_lda(A, l)
print(mu)
