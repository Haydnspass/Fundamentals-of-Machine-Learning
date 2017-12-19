import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy
from skimage.transform import resize
from scipy import optimize
from scipy import ndimage
from scipy.spatial.distance import cdist

def create_grid_for_image(data):
    xv, yv = np.meshgrid(range(data.shape[0]), range(data.shape[1]))
    points = np.zeros((xv.size,2))
    points[:,0] = xv.flatten()
    points[:,1] = yv.flatten()

    return points


def split_train_pred(grid, data):
    ix_pred = data == 0
    ix_train = np.logical_not(ix_pred)

    x_train = grid[ix_train,:]
    x_pred = grid[ix_pred,:]

    y_train = data[ix_train]
    y_pred = data[np.logical_not(ix_train)]

    return x_train, y_train, x_pred, y_pred

class Regression:
    def __init__(self, x_train, y_train):
        self.x = x_train
        self.y = y_train
        
    @property
    def n(self):
            return self.x.__len__()

    def kernel(self, x1, x2, kwidth=1):

        kernel = cdist(x1, x2, 'euclidean')
        kernel = np.exp(-(kernel ** 2) / (2. * kwidth ** 2))

        return kernel

        # n = self.x.__len__()
        # K_gauss = np.zeros((x1.__len__(),x2.__len__()))
        # for (i,j) in itertools.product(range(n), repeat = 2):
        #     K_gauss[i,j] = np.exp(-np.linalg.norm(x2[j,:] - x1[i,:]))
        #
        # return K_gauss
        
    def train(self, kwidth=1, llambda=1):

        K = self.kernel(self.x, self.x, kwidth=kwidth)
        alphas = np.dot(np.linalg.inv(K + llambda * np.eye(np.shape(K)[0])),self.y.transpose())
        self.alphas = alphas

    def predict(self, x_test):
        k = self.kernel(x_test, self.x)
        y_test = np.dot(k, self.alphas)
        return y_test.transpose()

if __name__ == '__main__':
    #load data
    img = ndimage.imread('cc_90.png', flatten=True).astype(np.int)
    x_t, y_t, x_p, y_p = split_train_pred(create_grid_for_image(img), img.flatten())


    imgC = Regression(x_t, y_t)# Regression(np.expand_dims(np.array(range(100)), axis=1), img)
    imgC.train()
    y_pred = imgC.predict(x_p)
    img_pred = img.__deepcopy__(img)
    img_pred[x_p[:,0].astype(np.int),x_p[:,1].astype(np.int)] = y_pred.astype(np.int)

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(img, cmap='gray')
    ax2.imshow(img_pred, cmap='gray')
    plt.show()
    print('Finished.')
