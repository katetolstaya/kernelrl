import numpy as np
import scipy.linalg as sl
from corerl.kernel import make_kernelN
import json

# Function class with preallocated matrices!

class KernelRepresentation(object):
    # N = dimension of state space
    # M = dimension of output space
    # K = number of dictionary elements
    def __init__(self, N, M, config):
        # Action space boundaries for clipping
        self.min_act = np.reshape(json.loads(config.get('MinAction')), (-1, 1))
        self.max_act = np.reshape(json.loads(config.get('MaxAction')), (-1, 1))

        # Argmax gradient routine parameters
        self.preallocate = config.getfloat('Preallocate', 300)
        self.grad_step = config.getfloat('GradStep', 0.05)
        self.grad_prec = config.getfloat('GradPrecision', 0.005)
        self.n_iters = config.getint('GradIters', 20)
        self.n_points = config.getint('GradPoints', 10)

        self.kernel = make_kernelN(config, N)
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
        # KDD is a K x K kernel matrix
        self.KDD = np.zeros((self.preallocate, self.preallocate))
        # U is the upper triangular decomposition of KDDinv
        self.U = np.zeros((self.preallocate, self.preallocate))

    # ------------------------------
    # Get model order
    # ------------------------------
    def model_order(self):
        return np.count_nonzero(self.idxs)

    # ------------------------------
    # Evaluate a set of values
    # ------------------------------
    # X : a L x N matrix where L is number of points to evaluate
    def __call__(self, X):
        # value = self.kernel.f(X, self.D).dot(self.W) + self.baseline
        value = self.kernel.f(X, self.D[self.idxs, :]).dot(self.W[self.idxs, :])
        return value

    # ------------------------------
    # Gradient with respect to (x,a)
    # ------------------------------
    def df(self, x):
        tempW = np.reshape(self.W[self.idxs,0], (1, 1, -1))
        tempdf = self.kernel.df(x, self.D[self.idxs, :])
        return np.reshape(np.dot(tempW, tempdf), np.shape(x))

    # ------------------------------
    # Argmax - given a state, find the optimal action using gradient ascent
    # ------------------------------
    def argmax(self, Y):
        # only supports 1 point in query!!!!
        Y = np.reshape(Y, (1, -1))
        dim_s2 = np.shape(Y)[1]  # num states
        dim_d1 = np.count_nonzero(self.idxs)  # n points in dictionary
        dim_d2 = self.N  # num states + actions

        if dim_d1 == 0:  # dictionary is empty
            cur_x = np.zeros((dim_d2 - dim_s2, 1))
            return cur_x

        # randomly generating action points
        acts = np.zeros((self.n_points, dim_d2))
        for i in range(0, dim_d2 - dim_s2):
            acts[:, i + dim_s2] = np.random.uniform(self.min_act[i], self.max_act[i], (self.n_points,))
        acts[:, 0:dim_s2] = np.tile(Y, (self.n_points, 1))

        # start gradient descent iterations
        keep_updating = np.full((self.n_points,), True, dtype=bool)
        iters = 0
        while (keep_updating.any()) and iters < self.n_iters:
            iters = iters + 1

            df = np.zeros((self.n_points, dim_d2))
            df[keep_updating, :] = self.df(acts[keep_updating, :])
            df[:, 0:dim_s2] = 0

            acts = acts + self.grad_step  * df

            temp1 = np.logical_and(np.any(acts[:, dim_s2:] <= self.max_act, axis=1),
                                   np.any(acts[:, dim_s2:] >= self.min_act, axis=1))
            temp2 = np.logical_and(temp1, np.linalg.norm(self.grad_step * df, axis=1) > self.grad_prec)
            keep_updating = temp2

        for i in range(0, dim_d2 - dim_s2):
            acts[:, i + dim_s2] = np.clip(acts[:, i + dim_s2], self.min_act[i], self.max_act[i])

        # break ties
        b = self(acts)[:,0]
        amax = np.array([np.argmax(np.random.random(b.shape) * (b == b.max()))])
        action = np.reshape(acts[amax, dim_s2:], (-1, 1))
        return action

    # ------------------------------
    # Helper function for indexing preallocated arrays
    # ------------------------------
    def get_new_idxs(self,n):
        new_idxs = list()
        for i in range(0, n):
            if len(self.unused) > 0:
                idx = int(self.unused.pop())
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
                    self.U.resize((self.preallocate, self.preallocate))
            new_idxs.append(idx)
        return new_idxs

    def push_unused(self,list_ind):
        self.unused.extend(list_ind)
        self.D[list_ind,:] *= 0
        self.W[list_ind,:] *= 0
        self.KDD[list_ind,:] *= 0
        self.KDD[:,list_ind] *= 0
        self.U[list_ind,:] *= 0
        self.U[:,list_ind] *= 0
    # ------------------------------
    # Shrink current dictionary weights
    # ------------------------------
    def shrink(self, s):
        self.W *= s

    # ------------------------------
    # Append new dictionary points and weights
    # ------------------------------
    def append(self, Dnew, Wnew):

        Nnew = len(Dnew)

        if len(Dnew.shape) == 1:
            Dnew = Dnew.reshape((1, Nnew))
        if len(Wnew.shape) == 1:
            Wnew = Wnew.reshape((1, Nnew))

        new_idxs = self.get_new_idxs(Nnew)

        # update kernel matrix
        self.KDD[np.ix_(new_idxs, new_idxs)] = self.kernel.f(Dnew, Dnew) + 1e-9 * np.eye(Nnew) # KXX
        self.KDD[np.ix_(self.idxs, new_idxs)] = self.kernel.f(self.D[self.idxs, :], Dnew) # KDX
        self.KDD[np.ix_(new_idxs, self.idxs)] = self.KDD[np.ix_(self.idxs, new_idxs)].T

        C12 = self.U[np.ix_(self.idxs, self.idxs)].T.dot(self.KDD[np.ix_(self.idxs, new_idxs)])
        C22 = sl.cholesky(self.KDD[np.ix_(new_idxs, new_idxs)] - C12.T.dot(C12))

        # update kernel matrix inverse decomposition
        self.U[np.ix_(new_idxs, new_idxs)] = sl.solve_triangular(C22, np.eye(Nnew), overwrite_b=True)
        self.U[np.ix_(self.idxs, new_idxs)] = -self.U[np.ix_(self.idxs, self.idxs)].dot(C12).dot(self.U[np.ix_(new_idxs, new_idxs)])
        self.U[np.ix_(new_idxs,self.idxs)] *= 0  #np.zeros((Nnew, np.count_nonzero(self.idxs)))#U12.T
        self.D[new_idxs, :] = Dnew
        self.W[new_idxs, :] = Wnew
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
        R = self.W[self.idxs,:].copy() #TODO preallocate
        # running computation of projection of dictionary elements
        V = self.U[np.ix_(self.idxs, self.idxs)].dot(self.U[np.ix_(self.idxs, self.idxs)].T)

        # the set of indices of D that we are keeping
        Y = self.idxs[self.idxs].copy()#self.idxs.copy()

        # remove points as long as we can
        while np.any(Y):
            # current error if we remove each point
            d = np.sum(R ** 2, axis=1) / np.diagonal(V)
            # find minimum
            Sy, y = S + np.min(d), np.argmin(d)
            # check error
            if Sy > eps2:
                break
            # updates
            S = Sy
            R -= np.outer(V[y], R[y]) / V[y, y]
            V -= np.outer(V[y], V[y]) / V[y, y]
            # correct possible numerical issues with future error calculations
            V[y, y] = 1.
            R[y] = 100.
            # remove point from model
            Y[y] = False

        if not Y.all():
            tY = self.idxs.copy()
            tY[self.idxs] = Y

            # project weights onto remaining indices
            self.W[tY, :] = V[np.ix_(Y, Y)].dot(self.KDD[np.ix_(tY, self.idxs)].dot(self.W[self.idxs, :]))

            # reclaim unused idxs
            self.push_unused(np.where(np.logical_and(np.logical_not(tY),self.idxs))[0])

            self.idxs = tY
            if np.any(self.idxs):
                self.U[np.ix_(self.idxs, self.idxs)] = sl.solve_triangular(sl.cholesky(self.KDD[np.ix_(self.idxs, self.idxs)]),
                                                                           np.eye(np.count_nonzero(self.idxs)), overwrite_b=True)

    # ------------------------------
    # Hilbert-norm of this function
    # ------------------------------
    def normsq(self):
        return self.W[self.idxs, :].T.dot(self.KDD[np.ix_(self.idxs, self.idxs)].dot(self.W[self.idxs, :])).trace()
