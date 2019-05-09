import numpy as np
import scipy.linalg as sl
from corerl.kernel import make_kernelN
import json
from rtree import index

# TODO- profile functions, misc on montain car

# Function class with tree index for points for pruning faster

class KernelRepresentation(object):
    # N = dimension of state space
    # M = dimension of output space
    # K = number of dictionary elements
    def __init__(self, N, M, config):
        self.kernel = make_kernelN(config, N)

        self.preallocate = 1000

        self.N = N
        self.M = M

        # initialize RTree
        p = index.Property()
        p.dimension = N
        self.idx = index.Index( properties=p)

        # initialize ID indexes
        self.unique_index = 0
        self.idxs = np.full((self.preallocate,), False, dtype=bool)
        self.new_idxs = np.full((self.preallocate,), False, dtype=bool)
        self.unused = list()
        self.new_idx = list()

        # D is a K x N matrix of dictionary elements
        self.D = np.zeros((self.preallocate, N))
        # W is a K x M matrix of weights
        self.W = np.zeros((self.preallocate, M))

        self.min_act = np.reshape(json.loads(config.get('MinAction')), (-1, 1))
        self.max_act = np.reshape(json.loads(config.get('MaxAction')), (-1, 1))

        self.baseline = config.getfloat('Baseline',0)
        self.grad_step = config.getfloat('GradStep', 0.05)
        self.grad_prec = config.getfloat('GradPrecision',0.005)

        self.width = np.array(json.loads(config.get('GaussianBandwidth')))[0:self.N]

    def model_order(self):
        return self.unique_index + len(self.unused)

    def points_union(self, X): #TODO clip to dimensions of space
        lower = np.min(X,axis=0) - self.width
        upper = np.max(X,axis=0) + self.width
        return np.append(lower,upper).tolist()

    def expand_point(self,point):  #TODO clip to dimensions of space
        return tuple(np.concatenate((point - self.width, point + self.width)).tolist())

    def get_neighbors(self,X):
        union = self.points_union(X)
        if union is None:
            return None
        hits = self.idx.intersection(tuple(union), objects=True)
        return list(set(list(set([hit.id for hit in hits]))))

    # ------------------------------
    # Evaluate a set of values
    # ------------------------------
    # X : a L x N matrix where L is number of points to evaluate
    def __call__(self, X, cached_neighbors=None):
        #if cached_neighbors is None:
        #    neighbors = self.get_neighbors(X) #list(set([hit.id for hit in hits]))
        #    if neighbors is None:
        #        return 0
        #else:
        #    neighbors = cached_neighbors
        #value = self.kernel.f(X, self.D[neighbors, :]).dot(self.W[neighbors])

        value = self.kernel.f(X, self.D[self.idxs,:]).dot(self.W[self.idxs]) + self.baseline
        return value

    def df(self, X, cached_neighbors=None):
        if cached_neighbors is None:
            neighbors = self.get_neighbors(X)
            if neighbors is None:
                return np.zeros(self.M, np.shape(X)[0])
        else:
            neighbors = cached_neighbors
        tempW = np.reshape(self.W[neighbors], (1,1,-1))
        tempdf = self.kernel.df(X, self.D[neighbors, :])
        #tempW = np.reshape(self.W[self.idxs], (1,1,-1))
        #tempdf = self.kernel.df(X, self.D[self.idxs, :])
        return np.reshape(np.dot(tempW, tempdf), np.shape(X))

    def hessian(self, X, cached_neighbors=None):
        if cached_neighbors is None:
            neighbors = self.get_neighbors(X)
            if neighbors is None:
                return np.zeros(self.M, np.shape(X)[0])
        else:
            neighbors = cached_neighbors
        tempW = np.reshape(self.W[neighbors], (1,1,1,-1))
        tempH = self.kernel.hessian(X, self.D[neighbors, :])
        #tempW = np.reshape(self.W[self.idxs], (1,1,-1))
        #tempdf = self.kernel.df(X, self.D[self.idxs, :])
        return np.reshape(np.dot(tempW, tempH), (self.N,self.N,np.shape(X)[0]))


    def argmax(self, Y):
        Y = np.reshape(Y, (1, -1))
        dim_s2 = np.shape(Y)[1]  # num states

        dim_d1 = np.sum(self.idxs) # n points in dictionary
        dim_d2 = self.N  # num states + actions

        if dim_d1 == 0:  # dictionary is empty
            cur_x = np.zeros((dim_d2 - dim_s2, 1))
            return cur_x

        # Gradient Descent Parameters
        gamma = self.grad_step # step size
        precision = self.grad_prec
        stop_iters = 0 #20

        #N = min(int(math.ceil(float(dim_d1)/3)),20)  # num points to misc
        N = min(max(dim_d1,2),20)

        acts = np.zeros((N, dim_d2))

        # randomly generating action points
        for i in range(0, dim_d2 - dim_s2):
            acts[:, i + dim_s2] = np.random.uniform(self.min_act[i], self.max_act[i], (N,))

        #acts[0, dim_s2:] = self.min_act
        #acts[1, dim_s2:] = self.max_act

        acts[:, 0:dim_s2] = np.tile(Y, (N, 1))

        iters = 0
        keep_updating = np.full((N,), True, dtype=bool)

        #neighbors = self.get_neighbors(acts)
        #btemp = self(acts)

        while (keep_updating.any()) and iters < stop_iters:
            iters = iters + 1

            df = np.zeros((N,dim_d2))
            df[keep_updating, :] = self.df(acts[keep_updating, :],None) #neighbors)
            #df[keep_updating, :] = self.df(acts[keep_updating, :],None)
            df[:, 0:dim_s2] = 0

            acts = acts + gamma * df

            temp1 = np.logical_and(np.any(acts[:,dim_s2:] <= self.max_act, axis=1), np.any(acts[:, dim_s2:] >= self.min_act, axis=1))
            temp2 = np.logical_and(temp1, np.linalg.norm(gamma * df, axis=1) > precision)
            keep_updating = temp2

        for i in range(0, dim_d2 - dim_s2):
            acts[:, i + dim_s2] = np.clip(acts[:,i + dim_s2], self.min_act[i], self.max_act[i])

        b = self(acts)

        # break ties
        amax = np.array([np.argmax(np.random.random(b.shape) * (b == b.max()))])
        action = np.reshape(acts[amax, dim_s2:], (-1, 1))
        return action

    def argmax_hessian(self, Y):  #doesn't work yet TODO
        Y = np.reshape(Y, (1, -1))
        dim_s2 = np.shape(Y)[1]  # num states

        dim_d1 = np.sum(self.idxs) # n points in dictionary
        dim_d2 = self.N  # num states + actions

        if dim_d1 == 0:  # dictionary is empty
            cur_x = np.zeros((dim_d2 - dim_s2, 1))
            return cur_x

        # Gradient Descent Parameters
        gamma = self.grad_step # step size
        precision = self.grad_prec
        stop_iters = 6

        #N = min(int(math.ceil(float(dim_d1)/3)),20)  # num points to misc
        N = min(max(dim_d1,2),20)

        acts = np.zeros((N, dim_d2))

        # randomly generating action points
        for i in range(0, dim_d2 - dim_s2):
            acts[:, i + dim_s2] = np.random.uniform(self.min_act[i], self.max_act[i], (N,))

        acts[0, dim_s2:] = self.min_act
        acts[1, dim_s2:] = self.max_act

        acts[:, 0:dim_s2] = np.tile(Y, (N, 1))

        iters = 0
        keep_updating = np.full((N,), True, dtype=bool)

        neighbors = self.get_neighbors(acts)

        df = np.zeros((N, dim_d2))
        hess = np.zeros((dim_d2, dim_d2, N))
        hess_inv = np.zeros((dim_d2, dim_d2, N))

        btemp = self(acts)
        while (keep_updating.any()) and iters < stop_iters:
            iters = iters + 1

            df[:] = 0
            hess[:] = 0
            hess_inv[:] = 0

            hess[:,:,keep_updating] = self.hessian(acts[keep_updating, :],neighbors)
            df[keep_updating, :] = self.df(acts[keep_updating, :],neighbors)

            #df[keep_updating, :] = self.df(acts[keep_updating, :],None)
            df[:, 0:dim_s2] = 0

            for i in range(0,len(keep_updating)):
                if keep_updating[i]:
                    #print hess[:,:,i]
                    try:
                        hess_inv[dim_s2:,dim_s2:,i] = np.reshape(np.linalg.inv(np.reshape(hess[:,:,i],(dim_d2,dim_d2))),(1,dim_d2,dim_d2))[:,dim_s2:,dim_s2:]

                    except np.linalg.linalg.LinAlgError as err:
                        if 'Singular matrix' in err.message:
                            print hess[:,:,i]
                        else:
                            raise

            dacts = np.sum(np.dot(df,hess_inv),axis=2)

            acts = acts + dacts # * gamma

            temp1 = np.logical_and(np.any(acts[:,dim_s2:] <= self.max_act, axis=1), np.any(acts[:, dim_s2:] >= self.min_act, axis=1))
            temp2 = np.logical_and(temp1, np.linalg.norm(dacts, axis=1) > precision)
            keep_updating = temp2

        #print iters
        #print np.max( np.linalg.norm(dacts, axis=1))
        #print dacts
        #print acts

        for i in range(0, dim_d2 - dim_s2):
            acts[:, i + dim_s2] = np.clip(acts[:,i + dim_s2], self.min_act[i], self.max_act[i])



        b = self(acts)
        print np.mean((b - btemp)/btemp)
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
    def append(self, Dnew, Wnew):
        for i in range(0,len(Dnew)):
            if len(self.unused) > 0:
                idx = self.unused.pop()
            else:
                idx = self.unique_index
                self.unique_index += 1

                # resize the data structures when out of room
                if self.unique_index >= self.preallocate:
                    self.preallocate = self.preallocate*2
                    self.idxs.resize((self.preallocate,))
                    self.new_idxs.resize((self.preallocate,))
                    self.D.resize((self.preallocate, self.N))
                    self.W.resize((self.preallocate, self.M))

            self.idxs[idx] = True
            self.new_idxs[idx] = True
            self.D[idx, :] = Dnew[i,:]

            #print np.shape(self.W[idx, :])
            #print np.shape(Wnew)
            self.W[idx, :] = Wnew[i,:]



    def new_region_union(self):
        if np.sum(self.new_idxs)==0:
            return None
        else:
            #Dtemp =  self.expand_point(self.D[self.new_idxs,:])
            #print np.shape(np.min(self.D[self.new_idxs,:],axis=0))
            #print np.shape(self.width[0:self.N])
            lower = np.min(self.D[self.new_idxs,:],axis=0) - self.width
            upper = np.max(self.D[self.new_idxs,:],axis=0) + self.width
            return np.append(lower,upper).tolist()

    # ------------------------------
    # Prune the current representation according to given approximation budget
    # ------------------------------
    # eps2 : allowed squared approximation error from current value
    #
    # NOTE: this function is not idempotent when called repeatedly with the same
    #       approximation budget since the current value is moved each time to be
    #       the 'simplest' value within an eps-ball

    def prune(self, eps2):

        # get and process neighbor kernel centers
        if np.sum(self.new_idxs) == 0:
            return



        neighbors = self.get_neighbors(self.D[self.new_idxs,:]) #list(set([hit.id for hit in hits]))

        #print str(len(neighbors)) + '/' + str(self.unique_index)
        if neighbors is None:
            neighbors = list()

        # check new points for approximate linear dependence
        lindep_thresh = 0.9999
        if len(neighbors) > 0:
            KDD_temp = self.kernel.f(self.D[self.new_idxs, :], self.D[neighbors, :])
            temp = np.max(KDD_temp, axis=1)
            new_nodes = (np.where(self.new_idxs)[0][np.where(temp < lindep_thresh)]).tolist()
            new_removed = (np.where(self.new_idxs)[0][np.where(temp >= lindep_thresh)]).tolist()
        else:
            new_nodes = np.where(self.new_idxs)[0].tolist()
            new_removed = list()
        neighbors.extend(new_nodes)

        # KOMP
        KDD = self.kernel.f(self.D[neighbors, :], self.D[neighbors, :]) + 1e-9 * np.eye(len(neighbors))
        U = sl.solve_triangular(sl.cholesky(KDD), np.eye(len(neighbors)))
        # running total of approximation error
        S = 0.
        # running computation of approximation residue
        R = self.W[neighbors,:].copy()
        # running computation of projection of dictionary elements
        V = U.dot(U.T)
        # the set of indices of D that we are keeping
        Y = np.full((len(neighbors),), True, dtype=bool)
        # remove points as long as we can
        while np.sum(Y) > 0:
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

        Yremoved = np.append(np.array(neighbors)[np.logical_not(Y)],new_removed).astype(int) # new and old removed, unique indexes
        Ykept = np.array(neighbors)[Y] # kept points, unique indexes

        # project weights onto remaining indices
        if len(Ykept) > 0 and len(Yremoved)>0:
            self.W[Ykept] = V[np.where(Y)[0],:].dot(KDD.dot(self.W[neighbors]))

        for j in Yremoved:
            if not self.new_idxs[j]: #if not new, remove from tree
                self.idx.delete(j,self.expand_point(self.D[j,:]) )
            self.unused.append(j)
            self.idxs[j] = False
        for j in Ykept:
            if self.new_idxs[j]: # if new, add to tree
                expanded_point = self.expand_point(self.D[j,:])
                self.idx.insert(j, expanded_point)

        self.new_idxs[:] = False

    # ------------------------------
    # Hilbert-norm of this function
    # ------------------------------
    def normsq(self): #TODO

         #KDD = self.kernel.f(self.D[self.idxs, :], self.D[self.idxs, :]) + 1e-9 * np.eye(np.sum(self.idxs))
         #return self.W[self.idxs].T.dot(KDD.dot(self.W[self.idxs])).trace()
        return 0
