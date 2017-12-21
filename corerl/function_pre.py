import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
from kernel import make_kernel
from kernel import make_kernelN
import json
import math


class KernelRepresentation(object):
    # N = dimension of state space
    # M = dimension of output space
    # K = number of dictionary elements
    def __init__(self, N, M, config):
        self.kernel = make_kernelN(config, N)

        self.preallocate = 300
        self.N = N
        self.M = M

        # initialize ID indexes
        self.unique_index = 0
        self.idxs = np.full((self.preallocate,), False, dtype=bool)
        self.unused = list()

        # D is a K x N matrix of dictionary elements
        self.D = np.zeros((self.preallocate, N))
        # W is a K x M matrix of weights
        self.W = np.zeros((self.preallocate, M))
        self.KDD = np.zeros((self.preallocate, self.preallocate))
        # U is the upper triangular decomposition of KDDinv
        self.U = np.zeros((self.preallocate, self.preallocate))

        self.min_act = np.reshape(json.loads(config.get('MinAction')), (-1, 1))
        self.max_act = np.reshape(json.loads(config.get('MaxAction')), (-1, 1))

        self.grad_step = config.getfloat('GradStep', 0.05)
        self.grad_prec = config.getfloat('GradPrecision', 0.005)
        self.baseline = config.getfloat('Baseline', 0)

def model_order(self):
    return len(self.W)

# ------------------------------
# Evaluate a set of values
# ------------------------------
# X : a L x N matrix where L is number of points to evaluate

def __call__(self, X):
    # value = self.kernel.f(X, self.D).dot(self.W) + self.baseline
    value = self.kernel.f(X, self.D[self.idxs, :]).dot(self.W[self.idxs, :])
    return value

def df(self, x):
    tempW = np.reshape(self.W[self.idxs], (1, 1, -1))
    tempdf = self.kernel.df(X, self.D[self.idxs, :])
    return np.reshape(np.dot(tempW, tempdf), np.shape(x))

def argmax(self, Y):
    Y = np.reshape(Y, (1, -1))
    dim_s2 = np.shape(Y)[1]  # num states

    dim_d1 = np.sum(self.idxs)  # n points in dictionary
    dim_d2 = self.N  # num states + actions

    if dim_d1 == 0:  # dictionary is empty
        cur_x = np.zeros((dim_d2 - dim_s2, 1))
        return cur_x

    # Gradient Descent Parameters
    gamma = self.grad_step  # step size
    precision = self.grad_prec
    stop_iters = 0  # 20

    # N = min(int(math.ceil(float(dim_d1)/3)),20)  # num points to test
    N = min(max(dim_d1, 2), 20)

    acts = np.zeros((N, dim_d2))

    # randomly generating action points
    for i in range(0, dim_d2 - dim_s2):
        acts[:, i + dim_s2] = np.random.uniform(self.min_act[i], self.max_act[i], (N,))

    # acts[0, dim_s2:] = self.min_act
    # acts[1, dim_s2:] = self.max_act

    acts[:, 0:dim_s2] = np.tile(Y, (N, 1))

    iters = 0
    keep_updating = np.full((N,), True, dtype=bool)

    # neighbors = self.get_neighbors(acts)
    # btemp = self(acts)

    while (keep_updating.any()) and iters < stop_iters:
        iters = iters + 1

        df = np.zeros((N, dim_d2))
        df[keep_updating, :] = self.df(acts[keep_updating, :], None)  # neighbors)
        # df[keep_updating, :] = self.df(acts[keep_updating, :],None)
        df[:, 0:dim_s2] = 0

        acts = acts + gamma * df

        temp1 = np.logical_and(np.any(acts[:, dim_s2:] <= self.max_act, axis=1),
                               np.any(acts[:, dim_s2:] >= self.min_act, axis=1))
        temp2 = np.logical_and(temp1, np.linalg.norm(gamma * df, axis=1) > precision)
        keep_updating = temp2

    for i in range(0, dim_d2 - dim_s2):
        acts[:, i + dim_s2] = np.clip(acts[:, i + dim_s2], self.min_act[i], self.max_act[i])

    b = self(acts)

    # break ties
    amax = np.array([np.argmax(np.random.random(b.shape) * (b == b.max()))])
    action = np.reshape(acts[amax, dim_s2:], (-1, 1))
    return action


# ------------------------------
# Shrink current dictionary weights
# ------------------------------
def shrink(self, s):
    self.W *= s


# ------------------------------
# Append new dictionary points and weights
# ------------------------------

def get_new_idxs(self,n):
    new_idxs = list()
    for i in range(0, n):
        if len(self.unused) > 0:
            idx = self.unused.pop()
        else:
            idx = self.unique_index
            self.unique_index += 1
            # resize the data structures when out of room
            if self.unique_index >= self.preallocate:
                self.preallocate = self.preallocate * 2
                self.idxs.resize((self.preallocate,))
                self.D.resize((self.preallocate, self.N))
                self.W.resize((self.preallocate, self.M))
                self.KDD.resize((self.preallocate, self.preallocate))
        new_idxs.append(idx)
    return new_idxs

def append(self, Dnew, Wnew):

    new_idxs = self.get_new_idxs(len(Dnew))

    if len(Dnew.shape) == 1:
        Dnew = Dnew.reshape((1, len(Dnew)))
    if len(Wnew.shape) == 1:
        Wnew = Wnew.reshape((1, len(Wnew)))
    # update kernel matrix
    KDX = self.kernel.f(self.D[self.idxs, :], Dnew)
    KXX = self.kernel.f(Dnew, Dnew) + 1e-9 * np.eye(len(Dnew))

    self.KDD[np.ix_(new_idxs, new_idxs)] = KXX
    self.KDD[np.ix_(new_idxs, self.idxs)] = KDX.T
    self.KDD[np.ix_(self.idxs, new_idxs)] = KDX

    # update kernel matrix inverse decomposition
    C12 = self.U[self.idxs, :].T.dot(KDX)
    C22 = sl.cholesky(KXX - C12.T.dot(C12))
    U22 = sl.solve_triangular(C22, np.eye(len(Dnew)), overwrite_b=True) #TODO - error here
    U12 = -self.U[self.idxs, :].dot(C12).dot(U22)

    self.U[np.ix_(new_idxs, new_idxs)] = U22
    self.U[np.ix_(self.idxs, new_idxs)] = U12
    # self.U[np.ix_(new_idxs,self.idxs)] = U12.T
    self.D[new_idxs, :] = Dnew  # np.concatenate((self.D, Dnew))
    self.W[new_idxs, :] = Wnew  # np.concatenate((self.W, Wnew),axis=0)
    self.idxs[new_idxs] = True

# ------------------------------
# Prune the current representation according to given approximation budget
# ------------------------------
# eps2 : allowed squared approximation error from current value
#
# NOTE: this function is not idempotent when called repeatedly with the same
#       approximation budget since the current value is moved each time to be
#       the 'simplest' value within an eps-ball
def prune(self, eps2):
    # running total of approximation error
    S = 0.
    # running computation of approximation residue
    self.R = self.W.copy()
    # running computation of projection of dictionary elements
    self.V = self.U.dot(self.U.T)
    # the set of indices of D that we are keeping
    Y = self.idxs.copy()
    # range(self.D.shape[0])
    # remove points as long as we can
    while np.any(Y):
        # current error if we remove each point
        # d = np.sum(R ** 2, axis=1) / np.diagonal(V)
        d = np.sum(self.R[Y, :] ** 2, axis=1) / np.diagonal(self.V[np.ix_(Y, Y)])
        # find minimum
        Sy = S + np.min(d)

        y = np.where(Y)[0][np.argmin(d)]
        # check error
        if Sy > eps2:
            break
        # updates
        S = Sy

        self.R[self.idxs, :] -= np.outer(self.V[y, self.idxs], self.R[y, :]) / self.V[y, y]
        self.V[np.ix_(self.idxs, self.idxs)] -= np.outer(self.V[y, self.idxs], self.V[y, self.idxs]) / self.V[y, y]
        # correct possible numerical issues with future error calculations
        self.V[y, y] = 1.
        self.R[y] = 100.
        # remove point from model
        Y[y] = False
    # project weights onto remaining indices
    self.W[Y, :] = self.V[np.ix_(Y, Y)].dot(self.KDD[np.ix_(Y, self.idxs)].dot(self.W[self.idxs, :]))
    self.idxs = Y
    if np.any(self.idxs):
        self.U[np.ix_(self.idxs, self.idxs)] = sl.solve_triangular(sl.cholesky(self.KDD[np.ix_(self.idxs, self.idxs)]),
                                                                   np.eye(np.sum(self.idxs)), overwrite_b=True)

# ------------------------------
# Hilbert-norm of this function
# ------------------------------
def normsq(self):
    return self.W[self.idxs, :].T.dot(self.KDD[np.ix_(self.idxs, self.idxs)].dot(self.W[self.idxs, :])).trace()
