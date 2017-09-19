import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
from kernel import make_kernel
from kernel import make_kernelN

class KernelRepresentation(object):
    # N = dimension of state space
    # M = dimension of output space
    # K = number of dictionary elements
    def __init__(self, N, M, config):
        self.kernel = make_kernelN(config,N)
        # D is a K x N matrix of dictionary elements
        self.D      = np.zeros((0,N))
        # KDD is a KxK matrix
        self.KDD    = np.zeros((0,0))
        # U is the upper triangular decomposition of KDDinv
        self.U      = np.zeros((0,0))
        # W is a K x M matrix of weights
        self.W      = np.zeros((0,M))
	
	self.low_act = np.array([-1])
        self.high_act = np.array([1])

    # ------------------------------
    # Evaluate a set of values
    # ------------------------------
    # X : a L x N matrix where L is number of points to evaluate
    def __call__(self, X):

        value = self.kernel.f(X, self.D).dot(self.W)

        return value
        # if len(X.shape) > 1 and X.shape[0] > 1:
        #     return value
        # else:
        #     if self.W.shape[1] > 1:
        #         return value[0]
        #     else:
        #         return value[0,0]

    def df (self, x):
	return self.kernel.df(x, self.D).dot(self.W)
    

    def argmax (self, Y):
	#print Y
        tempD = np.copy(self.D)
        dim1 = np.max(np.shape(Y))
        dim2 = self.D.shape[0]


        if dim2 == 0:
            cur_x =  np.zeros((self.D.shape[1]-dim1,1))
            return cur_x

        tempD[:, 0:dim1] = np.tile(np.reshape(Y, (1, -1)), (dim2, 1))
        b = self.kernel.f(tempD, self.D).dot(self.W)

	amax = np.array([np.argmax(np.random.random(b.shape) * (b==b.max()))])
        #amax = np.array([np.random.choice(np.flatnonzero(b == b.max()))]) 

        #amax = np.argmax(b)
	#print (amax)

        cur_x = np.reshape(tempD[amax,:],(-1,1))
	#print (cur_x)

        #gamma = 0.01  # step size multiplier
        #precision = 0.00001
	gamma = 0.05
        precision = 0.005
        stop_iters = 50
        iters = 0
        previous_step_size = np.linalg.norm(cur_x)


        while previous_step_size > precision and iters < stop_iters:
            iters = iters+1
            prev_x = cur_x
            df = self.kernel.df(prev_x, self.D).dot(self.W)
            df[0:dim1] = 0
            cur_x = cur_x + gamma * df
            previous_step_size = np.linalg.norm(cur_x - prev_x)
	    #print previous_step_size
	    if (cur_x[dim1:] <= self.low_act).any() or (cur_x[dim1:] >= self.high_act).any():
		break


        #if iters == stop_iters:
	#    print b
        #    print (previous_step_size)
        #print iters
	#self.state = np.clip(self.state, self.low_obs, self.high_obs)
	#print cur_x[dim1:]
	action = np.clip(cur_x[dim1:], self.low_act, self.high_act)
	#print action
	
	#if not np.array_equal(action, cur_x[dim1:]):
	#    print cur_x[dim1:]
	#    print action

        return action

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
        self.W *= s

    # ------------------------------
    # Append new dictionary points and weights
    # ------------------------------
    def append(self, Dnew, Wnew):
        if len(Dnew.shape) == 1:
            Dnew = Dnew.reshape((1, len(Dnew)))
        if len(Wnew.shape) == 1:
            Wnew = Wnew.reshape((1, len(Wnew)))
        # update kernel matrix
        KDX = self.kernel.f(self.D, Dnew)
        KXX = self.kernel.f(Dnew, Dnew) + 1e-9 * np.eye(len(Dnew))
        self.KDD = np.vstack((
            np.hstack((self.KDD, KDX)),
            np.hstack((KDX.T,    KXX))
            ))
        # update kernel matrix inverse decomposition
        C12 = self.U.T.dot(KDX)
        C22 = sl.cholesky(KXX - C12.T.dot(C12))
        U22 = sl.solve_triangular(C22, np.eye(len(Dnew)), overwrite_b=True)
        U12 = -self.U.dot(C12).dot(U22)
        self.U = np.vstack((
            np.hstack((self.U,                            U12)),
            np.hstack((np.zeros((len(Dnew),len(self.D))), U22))
            ))
        # check = self.U.dot(self.U.T).dot(self.KDD) - np.eye(len(self.KDD))
        # print 'inv check =', nl.norm(check)
        # if nl.norm(check) > 0.1:
        #     print 'bad!'
        #     print self.D
        #     print self.KDD
        self.D = np.concatenate((self.D, Dnew))
        self.W = np.concatenate((self.W, Wnew))

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
        R = self.W.copy()
        # running computation of projection of dictionary elements
        V = self.U.dot(self.U.T)
        # the set of indices of D that we are keeping
        Y = range(self.D.shape[0])
        # remove points as long as we can
        while len(Y) > 0:
            # current error if we remove each point
            d = np.sum(R**2, axis=1) / np.diagonal(V)
            # find minimum
            Sy, y = S + np.min(d), np.argmin(d)
            # check error
            if Sy > eps2:
                break
            # updates
            S = Sy
            R -= np.outer(V[y], R[y]) / V[y,y]
            V -= np.outer(V[y], V[y]) / V[y,y]
            # correct possible numerical issues with future error calculations
            V[y,y] = 1.
            R[y]   = 100.
            # remove point from model
            Y.remove(y)
        # project weights onto remaining indices
        self.W      = V[np.ix_(Y,Y)].dot(self.KDD[Y].dot(self.W))
        # adjust our dictionary elements
        self.D      = self.D[Y]
        self.KDD    = self.KDD[np.ix_(Y,Y)]
        if len(self.D) == 0:
            self.U = np.zeros((0,0))
        else:
            self.U = sl.solve_triangular(sl.cholesky(self.KDD), np.eye(len(Y)), overwrite_b=True)

    # ------------------------------
    # Hilbert-norm of this function
    # ------------------------------
    def normsq(self):
        return self.W.T.dot(self.KDD.dot(self.W)).trace()
