import numpy as np
import json


# ==================================================
# KERNELS
# --------------------------------------------------

def _distEucSq(s, X, Y):
    if len(X.shape) > 1:
        if len(Y.shape) > 1:
            m = X.shape[0]
            n = Y.shape[0]
            # print(Y.shape)
            # print(X.shape)
            XX = np.sum(X * s * X, axis=1)
            YY = np.sum(Y * s * Y, axis=1)
            # print(XX.shape)
            # print(YY.shape)
            return np.tile(XX.reshape(m, 1), (1, n)) + np.tile(YY, (m, 1)) - 2 * X.dot((s * Y).T)
        else:
            m = X.shape[0]
            XX = np.sum(X * s * X, axis=1)
            YY = np.sum(Y * s * Y)
            return np.tile(XX.reshape(m, 1), (1, 1)) + np.tile(YY, (m, 1)) - 2 * X.dot(s * Y).reshape(m, 1)
    else:
        if len(Y.shape) > 1:
            n = Y.shape[0]
            XX = np.sum(X * s * X)
            YY = np.sum(Y * s * Y, axis=1)
            return np.tile(XX, (1, n)) + np.tile(YY, (1, 1)) - 2 * X.dot((s * Y).T)
        else:
            m = 1
            n = 1
            XX = np.sum(X * s * X)
            YY = np.sum(Y * s * Y)
            return np.tile(XX, (1, 1)) + np.tile(YY, (1, 1)) - 2 * X.dot(Y)


class GaussianKernel(object):
    def __init__(self, sigma):
        self.sig = np.array(sigma)
        self.s = -1. / (2 * self.sig ** 2)

    def f(self, X, Y):

        sx1 = np.shape(X)[0]
        #sx2 = np.shape(X)[1]
        sy1 = np.shape(Y)[0]
        #sy2 = np.shape(Y)[1]

        ret = np.zeros((sx1,sy1))
        try:
            ret = np.exp(_distEucSq(self.s, X, Y))
        except AttributeError:
            print ("Attribute Error")
            print self.s
            print X
            print Y
        except ValueError:
            print "Value Error"
            print self.s
            print np.shape(X)
            print np.shape(Y)

        return ret

    def df2(self, Y, X):
        # assuming X is KxN and Y is 1xN
        K = np.shape(X)[0]
        N = np.shape(X)[1]
        Y = np.reshape(Y, (1, N))
        return (np.tile(self.f(X, Y), (1, N)) * np.tile(self.s, (K, 1)) * (np.tile(Y, (K, 1)) - X)).T

    def df(self,Y,X):
        # assume X is KxN and Y is MxN
        X1 = np.shape(X)[0]
        X2 = np.shape(X)[1]
        Y1 = np.shape(Y)[0]
        Y2 = np.shape(Y)[1]

        Xtemp = np.tile(np.reshape(X.T, (X2, X1, 1)), (1, 1, Y1))
        Ytemp = np.tile(np.reshape(Y.T, (Y2, 1, Y1)), (1, X1, 1))

        stemp = np.tile(np.reshape(self.s,(-1,1,1)), (1, X1, Y1))
        ftemp = np.tile(np.reshape(self.f(X, Y),(1,X1,Y1)), (X2,1,1))

        return 2 * ftemp * stemp * (Ytemp - Xtemp)

    def hessian(self, Y, X):
        # assume X is KxN and Y is MxN
        X1 = np.shape(X)[0]
        X2 = np.shape(X)[1]
        Y1 = np.shape(Y)[0]
        Y2 = np.shape(Y)[1]

        XYtemp = np.tile(np.reshape(X.T, (X2, 1, X1, 1)), (1, X2, 1, Y1)) - np.tile(np.reshape(Y.T, (Y2, 1, 1, Y1)), (1, X2, X1, 1))
        XY = XYtemp * np.swapaxes(XYtemp,0,1)

        stemp = np.tile(np.reshape(self.s, (-1, 1, 1, 1)), (1, X2, X1, Y1)) * np.tile(np.reshape(self.s, (1, -1, 1, 1)), (X2, 1, X1, Y1))
        ftemp = np.tile(np.reshape(self.f(X, Y), (1,1,X1,Y1)), (X2,X2,1,1))
        eye = np.tile(np.reshape(np.eye(X2),(X2,X2,1,1)),(1,1,X1,Y1))

        return 4 * ftemp * stemp**2 * XY + 2*eye*stemp*ftemp

    # --------------------------------------------------

def make_kernel(config):
    if config.get('KernelType', fallback='Gaussian').lower() == 'gaussian':
        return GaussianKernel(json.loads(config.get('GaussianBandwidth')))
    else:
        raise ValueError('Unknown kernel type: %s' % config.get('KernelType'))


def make_kernelN(config, n):
    if config.get('KernelType', fallback='Gaussian').lower() == 'gaussian':
        return GaussianKernel(json.loads(config.get('GaussianBandwidth'))[0:n])
    else:
        raise ValueError('Unknown kernel type: %s' % config.get('KernelType'))
