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

        # D is a K x N matrix of dictionary elements
        self.D = np.zeros((0, N))
        # KDD is a KxK matrix
        self.KDD = np.zeros((0, 0))
        # U is the upper triangular decomposition of KDDinv
        self.U = np.zeros((0, 0))
        # W is a K x M matrix of weights
        self.W = np.zeros((0, M))

        self.low_act = np.reshape(json.loads(config.get('MinAction')), (-1, 1))
        self.high_act = np.reshape(json.loads(config.get('MaxAction')), (-1, 1))

        self.grad_step = config.getfloat('GradStep', 0.05)
        self.grad_prec = config.getfloat('GradPrecision',0.005)
        self.baseline = config.getfloat('Baseline', 0)

        self.divergence = False

    def model_order(self):
        return len(self.W)

    # ------------------------------
    # Evaluate a set of values
    # ------------------------------
    # X : a L x N matrix where L is number of points to evaluate
    def __call__(self, X):
        if self.divergence:
            return np.zeros((self.D.shape[1],1))
        value = self.kernel.f(X, self.D).dot(self.W) + self.baseline
        return value

    def df(self, x):
        if self.divergence:
            return np.zeros(np.shape(x))

        tempW = np.reshape(self.W[:,0], (1,1,-1)) # use only axis 0 for argmax, df
        tempdf = self.kernel.df(x,self.D)
        return  np.reshape(np.dot(tempW, tempdf),np.shape(x))

    def argmax(self, Y):


        # only support 1 point in query!!!!
        Y = np.reshape(Y, (1, -1))
        dim_s2 = np.shape(Y)[1]  # num states

        dim_d1 = self.D.shape[0]  # n points in dictionary
        dim_d2 = self.D.shape[1]  # num states + actions

        if self.divergence:
            return np.zeros((dim_d2 - dim_s2, 1))


        if dim_d1 == 0:  # dictionary is empty
            cur_x = np.zeros((dim_d2 - dim_s2, 1))
            return cur_x

        # GRADIENT DESCENT PARAMS
        gamma = self.grad_step # step size
        precision = self.grad_prec
        stop_iters = 50

        N = min(dim_d1,20)
        acts = np.zeros((N, dim_d2))

        # randomly generating action points
        for i in range(0, dim_d2 - dim_s2):
            acts[:, i + dim_s2] = np.random.uniform(self.low_act[i], self.high_act[i], (N,))

        acts[:, 0:dim_s2] = np.tile(Y, (N, 1))

        iters = 0
        keep_updating = np.full((N,), True, dtype=bool)

        #btemp = self(acts)
        while (keep_updating.any()) and iters < stop_iters:
            iters = iters + 1

            df = np.zeros((N,dim_d2))
            df[keep_updating, :] = self.df(acts[keep_updating, :])
            df[:, 0:dim_s2] = 0

            acts = acts + gamma * df

            temp1 = np.logical_and(np.any(acts[:,dim_s2:] <= self.high_act.T,axis=1), np.any(acts[:,dim_s2:] >= self.low_act.T,axis=1))
            temp2 = np.logical_and(temp1, np.linalg.norm(gamma * df, axis=1) > precision)

            keep_updating = temp2

        for i in range(0, dim_d2 - dim_s2):
            acts[:, i + dim_s2] = np.clip(acts[:,i + dim_s2], self.low_act[i], self.high_act[i])

        b = self(acts)[:,0]
        amax = np.array([np.argmax(np.random.random(b.shape) * (b == b.max()))])
        action = np.reshape(acts[amax, dim_s2:], (-1, 1))
        return action

    # def argmax2(self, Y):
    #     # print Y
    #     tempD = np.copy(self.D)
    #     dim1 = np.max(np.shape(Y))
    #     dim2 = self.D.shape[0]
    #
    #     if dim2 == 0:
    #         cur_x = np.zeros((self.D.shape[1] - dim1, 1))
    #         return cur_x
    #
    #     tempD[:, 0:dim1] = np.tile(np.reshape(Y, (1, -1)), (dim2, 1))
    #     b = self.kernel.f(tempD, self.D).dot(self.W)
    #
    #     amax = np.array([np.argmax(np.random.random(b.shape) * (b == b.max()))])
    #     cur_x = np.reshape(tempD[amax, :], (-1, 1))
    #
    #     gamma = 0.05  # step size
    #     precision = 0.0005
    #     stop_iters = 50
    #     iters = 0
    #     previous_step_size = np.linalg.norm(cur_x)
    #
    #     while previous_step_size > precision and iters < stop_iters:
    #         iters = iters + 1
    #         prev_x = cur_x
    #         df = self.df(prev_x.T)
    #
    #         df[0:dim1] = 0
    #         cur_x = cur_x + gamma * df.T
    #         previous_step_size = np.linalg.norm(cur_x - prev_x)
    #         # print previous_step_size
    #         if (cur_x[dim1:] <= self.low_act).any() or (cur_x[dim1:] >= self.high_act).any():
    #             break
    #
    #     action = np.clip(cur_x[dim1:], self.low_act, self.high_act)
    #
    #     return action

        # def actmax (self,Y,sN,aN, M):
        #     sampleN = M*10
        #     x = np.linspace(-M, M, sampleN)
        #     y = np.linspace(-M, M, sampleN)
        #     xv, yv = np.meshgrid(x, y)
        #
        #     temp = np.zeros((sampleN*sampleN, sN+aN))
        #     dim1 = np.max(np.shape(Y))
        #     temp[:, 0:dim1] = np.tile(np.reshape(Y, (1, -1)), (sampleN*sampleN, 1))
        #     temp[:,dim1:dim1+1] = np.reshape(xv,(-1,1))
        #     temp[:,dim1+1:dim1+2] = np.reshape(yv,(-1,1))

        #     value = self.kernel.f(temp, self.D).dot(self.W)
        #     amax = np.argmax(value)
        #
        #     return temp[amax, dim1:]

    # ------------------------------
    # Shrink current dictionary weights
    # ------------------------------
    def shrink(self, s):
        if self.divergence:
            return
        self.W *= s

    # ------------------------------
    # Append new dictionary points and weights
    # ------------------------------
    def append(self, Dnew, Wnew):

        if self.divergence:
            return


        if len(Dnew.shape) == 1:
            Dnew = Dnew.reshape((1, len(Dnew)))
        if len(Wnew.shape) == 1:
            Wnew = Wnew.reshape((1, len(Wnew)))
        # update kernel matrix
        KDX = self.kernel.f(self.D, Dnew)
        lindep_thresh = 0.99

        #new_removed = np.any(np.where(KDX >= lindep_thresh))
        #if new_removed:
        #    print 'LINDEP'

        KXX = self.kernel.f(Dnew, Dnew) + 1e-9 * np.eye(len(Dnew))
        self.KDD = np.vstack((
            np.hstack((self.KDD, KDX)),
            np.hstack((KDX.T, KXX))
        ))
        # update kernel matrix inverse decomposition
        C12 = self.U.T.dot(KDX)
        C22 = sl.cholesky(KXX - C12.T.dot(C12))
        U22 = sl.solve_triangular(C22, np.eye(len(Dnew)), overwrite_b=True)
        U12 = -self.U.dot(C12).dot(U22)
        self.U = np.vstack((
            np.hstack((self.U, U12)),
            np.hstack((np.zeros((len(Dnew), len(self.D))), U22))
        ))
        # check = self.U.dot(self.U.T).dot(self.KDD) - np.eye(len(self.KDD))
        # print 'inv check =', nl.norm(check)
        # if nl.norm(check) > 0.1:
        #     print 'bad!'
        #     print self.D
        #     print self.KDD
        self.D = np.concatenate((self.D, Dnew))
        self.W = np.concatenate((self.W, Wnew),axis=0)

    # ------------------------------
    # Prune the current representation according to given approximation budget
    # ------------------------------
    # eps2 : allowed squared approximation error from current value
    #
    # NOTE: this function is not idempotent when called repeatedly with the same
    #       approximation budget since the current value is moved each time to be
    #       the 'simplest' value within an eps-ball
    def prune(self, eps2):
        if self.divergence:
            return
        # running total of approximation error
        S = 0.
        # running computation of approximation residue
        R = self.W.copy()
        # running computation of projection of dictionary elements
        V = self.U.dot(self.U.T)
        # the set of indices of D that we are keeping
        Y = list(range(self.D.shape[0]))
        # remove points as long as we can
        while len(Y) > 0:
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
            try:
                Y.remove(y)
            except ValueError:
                print ('Divergence!!!!!!!!!!!!!!!!!')
                self.divergence = True
                break
        # project weights onto remaining indices
        self.W = V[np.ix_(Y, Y)].dot(self.KDD[Y].dot(self.W))
        # adjust our dictionary elements
        self.D = self.D[Y]
        self.KDD = self.KDD[np.ix_(Y, Y)]
        if len(self.D) == 0:
            self.U = np.zeros((0, 0))
        else:
            self.U = sl.solve_triangular(sl.cholesky(self.KDD), np.eye(len(Y)), overwrite_b=True)

    # ------------------------------
    # Hilbert-norm of this function
    # ------------------------------
    def normsq(self):
        if self.divergence:
            return 0
        return self.W.T.dot(self.KDD.dot(self.W)).trace()
