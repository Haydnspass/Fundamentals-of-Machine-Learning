import numpy as np
from scipy import sparse

def construct_x(M, a, h, alphas, Np=None):
    def coordinate_array(a, h, M):  # a: 2 element vector of coordinate origin, h: pixel distance
        if a.__class__ == list:
            a = np.array((a), ndmin=2).reshape((2,-1))
        # c = np.zeros((2, np.square(M)))
        c = a + np.arange(0, np.square(M)) * np.ones_like(a) * h
        return c

    def rot_unit_n(phi):
        return np.array([np.sin(phi) np.cos(phi)], ndim=2).reshape((2, -1))

    def projection(n, C, s0):
        p = np.dot(np.transpose(n), C) + s0
        return p

    # use sufficient number of sensor bins
    if Np is None:
        Np = np.ceil(np.sqrt(2) * M)

    c = coordinate_array(a, h, M)

    X = sparse.coo_matrix((weights, (i_ix, j_ix)), shape=(N, D), dtype=np.float32)


"""
X:      projection matrix
                X x b = y
b:      flattened mu
y:      response vector  / sinogram

M:      resolution along dimension in image (i.e. 256 x 256 px image)
D:      D = M x M

N_0:    number of projection angles
N_p:    number of bins
N:      response vector size N = N_0 x N_p

"""

if __name__ == '__main__':
    construct_X(10, [2, 25], 3, [0, 25, 75])