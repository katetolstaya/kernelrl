import numpy
import json

# ==================================================
# KERNELS
# --------------------------------------------------

def _distEucSq(s,X,Y):

    if len(X.shape) > 1:
        if len(Y.shape) > 1:
            m = X.shape[0]
            n = Y.shape[0]
	    #print(Y.shape)
	    #print(X.shape)
            XX = numpy.sum(X * s * X, axis=1)
            YY = numpy.sum(Y * s * Y, axis=1)
	    #print(XX.shape)
	    #print(YY.shape)
            return numpy.tile(XX.reshape(m,1),(1,n)) + numpy.tile(YY,(m,1)) - 2*X.dot((s * Y).T)
        else:
            m = X.shape[0]
            XX = numpy.sum(X * s * X, axis=1)
            YY = numpy.sum(Y * s * Y)
            return numpy.tile(XX.reshape(m,1),(1,1)) + numpy.tile(YY,(m,1)) - 2*X.dot(s * Y).reshape(m,1)
    else:
        if len(Y.shape) > 1:
            n = Y.shape[0]
            XX = numpy.sum(X * s * X)
            YY = numpy.sum(Y * s * Y, axis=1)
            return numpy.tile(XX,(1,n)) + numpy.tile(YY,(1,1)) - 2*X.dot((s * Y).T)
        else:
            m = 1
            n = 1
            XX = numpy.sum(X * s * X)
            YY = numpy.sum(Y * s * Y)
            return numpy.tile(XX,(1,1)) + numpy.tile(YY,(1,1)) - 2*X.dot(Y)

class GaussianKernel(object):
    def __init__(self, sigma):
        self.sig = numpy.array(sigma)
        self.s = -1. / (2 * self.sig**2)
    def f(self, X, Y):
	ret = 0
	try:
	    ret = numpy.exp(_distEucSq(self.s, X, Y))
	except AttributeError:
	    print ("Attribute Error")
	    print self.s
	    print X
            print Y
	except:
	    print "Unknown error" 
        return ret

    def df(self,Y,X):
        #assuming X is KxN and Y is 1xN
        K = numpy.shape(X)[0]
        N = numpy.shape(X)[1]
        Y = numpy.reshape(Y,(1,N))

        return (numpy.tile(self.f(X,Y),(1,N)) * numpy.tile(self.s,(K,1)) * (numpy.tile(Y,(K,1))-X)).T


# --------------------------------------------------

def make_kernel(config):
    if config.get('KernelType', fallback='Gaussian').lower() == 'gaussian':
        return GaussianKernel(json.loads(config.get('GaussianBandwidth')))
    else:
        raise ValueError('Unknown kernel type: %s' % config.get('KernelType'))


def make_kernelN(config,n):
    if config.get('KernelType', fallback='Gaussian').lower() == 'gaussian':
        return GaussianKernel(json.loads(config.get('GaussianBandwidth'))[0:n])
    else:
        raise ValueError('Unknown kernel type: %s' % config.get('KernelType'))
