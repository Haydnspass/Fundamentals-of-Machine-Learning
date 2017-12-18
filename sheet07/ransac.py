import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from circle_fit import fit_circle

def plot_sample_data(data):
    plt.scatter(data[:,0], data[:,1])

class Circle:
    def __init__(self, data):
        self.data = data
        self.x = np.expand_dims(data[:,0], axis=1)
        self.y = np.expand_dims(data[:,1], axis=1)

    def fit_model(self):
        h = np.concatenate((self.x,self.y,np.ones_like(self.x)),1)
        if np.linalg.matrix_rank(h) == h.shape[1]:
            a = np.linalg.lstsq(np.concatenate((self.x,self.y,np.ones_like(self.x)),1), -(self.x**2 + self.y**2))[0]
        else:
            self.center = None
            self.R = None
            return

        xc = -.5 * a[0]
        yc = -.5 * a[1]

        self.center = np.concatenate((xc,yc),0)
        self.R = np.sqrt((a[0]**2 + a[1]**2) / 4 - a[2])
        # self.center, self.R = fit_circle(self.data)

    def error_model(self, p):
        if self.R == None or (self.center == None).any():
            return np.inf
        else:
            distance = np.abs(np.linalg.norm(p-self.center) - self.R)
            return distance/self.R


def ransac(data, n, k, t, d):

    def setdiff2d(A,B):
        nrows, ncols = A.shape
        dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
                 'formats': ncols * [A.dtype]}

        C = np.setdiff1d(A.view(dtype), B.view(dtype))

        # This last bit is optional if you're okay with "C" being a structured array...
        return C.view(A.dtype).reshape(-1, ncols)
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
    for i in range(k):
        ixInlier = np.random.choice(np.arange(data.shape[0]), n)
        maybeinliers = data[ixInlier,:]
        maybemodel = Circle(maybeinliers)
        maybemodel.fit_model()
        if maybemodel.R == None:
            continue
        alsoinliers = []
        for p in setdiff2d(data, maybeinliers):
            if maybemodel.error_model(p) < t:
                alsoinliers.append(p)
        
        if alsoinliers.__len__() > d:
            # good model, but how good
            bettermodel = Circle(np.concatenate((maybeinliers, alsoinliers),0))
            bettermodel.fit_model()
            thiserr = bettermodel.error_model(p)
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
    return bestfit


if __name__ == '__main__':
    data = np.load('circles.npy')

    # plot_sample_data(data), plt.show()
    #data = np.array([[-1,0],[0,1],[1,0],[999,23587],[0,1.0001],[1,0.0001]])
    bestfit = ransac(data, n=3, k=100, t=0.2, d=10)

    plot_sample_data(data)
    circle = plt.Circle(bestfit.center, radius=bestfit.R, fill=False)
    plt.gca().add_patch(circle)
    plt.show()
