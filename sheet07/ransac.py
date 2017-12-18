import itertools
import numpy as np
import matplotlib.pyplot as plt
from ellipses import LsqEllipse

def plot_sample_data(data):
    plt.scatter(data[:,0], data[:,1])


def fit_model(data, weights):
    A = np.concatenate((a, np.ones_like(data[0,:]))).transpose
    y = np.sum(A ** 2, 1).transpose

    # mat product
    p = (A.transpose * np.diag(weights) * A) * np.invert(A).transpose * np.diag(weights) * y

    center = .5 * p[:-2,:]
    radius = np.sqrt(p[-1,1] + center.transpose * center)

    return center, radius


def error_model(p, model):
    # sum of squares


def ransac(data, model, n, k, t, d):
    """
    data – a set of observed data points
    model – a model that can be fitted to data points
    n – minimum number of data points required to fit the model
    k – maximum number of iterations allowed in the algorithm
    t – threshold value to determine when a data point fits a model
    d – number of close data points required to assert that a model fits well to data

    Return:
    bestfit – model parameters which best fit the data (or nul if no good model is found)
    """
    
    iterations = 0
    bestfit = None
    besterr = np.inf
    while iterations < k:
        maybeinliers = np.random.choice(data, n)
        maybemodel = fit_model(maybeinliers)
        alsoinliers = []
        for p in data[data not in maybeinliers]:
            if error_model(p, maybemodel) < t:
                alsoinliers.append(p)
        
        if alsoinliers.__len__ > d:
            # good model, but how good
            bettermodel = fit_model(np.concatenate(maybeinliers, alsoinliers))
            thiserr = error_model(p, model)
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
    return bestfit

if __name__ == '__main__':
    data = np.load('circles.npy')

    # plot_sample_data(data), plt.show()

