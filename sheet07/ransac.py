import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_sample_data(data):
    plt.scatter(data[:,0], data[:,1])


def fit_model(data, weights):
    x = np.expand_dims(data[:,0], axis=1)
    y = np.expand_dims(data[:,1], axis=1)
    a = np.linalg.solve(np.concatenate((x,y,np.ones_like(x)),1), -(x**2 + y**2))
    
    xc = -.5 * a[1]
    yc = -.5 * a[2]
    
    center = np.concatenate((xc,yc),0)
    
    R = np.sqrt((a[0]**2 + a[1]**2) / 4 - a[2])
    
    return R, center, a


def error_model(p, model):
    # sum of squares
    pass


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
    data = np.array([[-1,0],[0,1],[1,0]])
    R, center, a = fit_model(data, weights=np.ones_like(data[:,0]))
    plot_sample_data(data)
    circle1 = plt.Circle(center, R)
    plt.gcf().gca().add_artist(circle1)
    plt.show()
