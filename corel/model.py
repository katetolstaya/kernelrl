import copy, json, random
import numpy as np
from corerl.core import ScheduledParameter
from corerl.function import KernelRepresentation
import pdb
import pickle
import csv
import matplotlib.pyplot as plt
from enum import Enum

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser

fname = 'cfg/kadam2.cfg'

config = ConfigParser()
with open(fname, 'r') as f:
    config.read_file(f)
    
config = config[config.default_section]

class ModelParameters():
    # Learning rates
    eta = 0.4 #0.1
    beta1 = 0.4
    beta2 = 0.2
    # Regularization
    lossL =  1e-6
    # Representation error budget
    eps = 1.0

    dt = 100
    #mu = 455.0
    #sigma = 200
    mu = 0.0
    sigma = 100 #200 #100 #50

class GradType(Enum):
    SGD = 0
    MOM = 1
    VAR = 2
    MOMENTUM = 3
    DELTA = 4

class Model:


    def __init__(self, indim, outdim, grad_type):
        self.f = KernelRepresentation(indim, outdim, config)
        self.indim = indim
        self.outdim = outdim
        self.grad_type = grad_type
        self.delta = 0

    def model_error(self):
        return 0.5 * ModelParameters.lossL * self.f.normsq()

    def predict(self, x):  # Predict the Q function values for a batch of states.
        return self.f(x)

    def predictOne(self, x):  # Predict the Q function values for a single state.
        return self.f(np.reshape(x,(1,-1)))[0]

    def val(self, x):
        return self.predictOne(x)[0] + ModelParameters.mu

    def mom(self, x):
        return self.predictOne(x)[1]
        
    def var(self, x):
        return max(0, self.predictOne(x)[2]  + ModelParameters.sigma)

    def train(self, sample):

        x,y = sample
        grad = (self.val(x)-y)
        grad_sq = grad**2
         
        # V gradient
        W = np.zeros((3,))
        
        if self.grad_type ==GradType.SGD: # simple SGD
            W[0] =  -ModelParameters.eta * grad
        elif self.grad_type == GradType.MOM:
            W[0] =  -ModelParameters.eta * grad / (np.sqrt(self.var(x) + ModelParameters.eta**2))
            W[2] = (-ModelParameters.beta2 * self.var(x) + ModelParameters.beta2 * grad_sq)
        elif self.grad_type == GradType.VAR:
            grad_var = (grad-self.mom(x))**2
            #print(grad_var)
            W[0] =  -ModelParameters.eta * self.mom(x) / (np.sqrt(self.var(x) + ModelParameters.eta**2)) #/ np.sqrt(self.var(x) + ModelParameters.eta) 
            W[1] = (-ModelParameters.beta1 * self.mom(x) + ModelParameters.beta1 * grad) 
            W[2] = (-ModelParameters.beta2 * self.var(x) + ModelParameters.beta2 * grad_var) 
        elif self.grad_type == GradType.MOMENTUM:
            W[0] =  -ModelParameters.eta * self.mom(x) / (np.sqrt(self.var(x) + ModelParameters.eta**2))
            W[1] = (-ModelParameters.beta1 * self.mom(x) + ModelParameters.beta1 * grad) 
            W[2] = (-ModelParameters.beta2 * self.var(x) + ModelParameters.beta2 * grad_sq) 
        elif self.grad_type == GradType.DELTA:
            W[0] =  -ModelParameters.eta * self.delta
            self.delta = self.delta + (-ModelParameters.beta1 * self.delta + ModelParameters.beta1 * grad) 
        else:
            print('error')
        
        # Gradient step
        self.f.shrink(1. - ModelParameters.lossL)
        self.f.append(np.array(x), np.reshape(W, (1, -1)))
        # Prune
        self.f.prune(ModelParameters.eps)
        
        return (grad_sq/2, len(self.f.D))
        
    def loss(self, sample):
        x, y = sample
        return 0.5 * (self.val(x)-y)**2

    def point_density(self, x):
        return np.sum(self.f.kernel.f(x, self.f.D))
        
    def compose(f1, f2): #static function

        #f = KernelRepresentation(4, 3, config)
        f = Model(f1.indim, f1.outdim, f1.grad_type )
        d = np.vstack([f1.D, f2.D]) 
        
        thresh = np.shape(f1.D)[0]
        
        W = np.zeros((3,))
        for i in np.random.permutation(np.shape(d)[0]):
            x = d[i,:]
            if f.grad_type == GradType.SGD:
                if f1.point_density(x) >= f2.point_density(x) and i < thresh: # reciprocal here so larger is better
                    W[1] = -f.mom(x) + f1.mom(x)
                    W[2] = -f.var(x) + f1.var(x)
                    W[0] = -f.val(x) + f1.val(x)
                    f.f.append(np.array(x), np.reshape(W, (1, -1)))
                elif f1.point_density(x) < f2.point_density(x) and i >= thresh:
                    W[1] = -f.mom(x) + f2.mom(x)
                    W[2] = -f.var(x) + f2.var(x)
                    W[0] = -f.val(x) + f2.val(x)
                    f.f.append(np.array(x), np.reshape(W, (1, -1)))
            elif f.grad_type == GradType.MOM or f.grad_type == GradType.VAR:
                if f1.var(x) <= f2.var(x) and i < thresh: # reciprocal here so larger is better
                    W[1] = -f.mom(x) + f1.mom(x)
                    W[2] = -f.var(x) + f1.var(x)
                    W[0] = -f.val(x) + f1.val(x)
                    f.f.append(np.array(x), np.reshape(W, (1, -1)))
                elif f1.var(x) > f2.var(x) and i >= thresh:
                    W[1] = -f.mom(x) + f2.mom(x)
                    W[2] = -f.var(x) + f2.var(x)
                    W[0] = -f.val(x) + f2.val(x)
                    f.f.append(np.array(x), np.reshape(W, (1, -1)))
        return f

def train_model(reader, grad_type=GradType.MOM):
    f = Model(4, 3, grad_type)

    losses = []

    for i, row in enumerate(reader):
        sample = (np.array([float(row[0]), float(row[1]), float(row[2]), float(row[3])]), float(row[4]))
        (loss, model) =  f.train(sample)
        losses.append(loss)        
    return f
    
def test_model(f, reader):
    n = 0
    temp_loss = 0
    for i, row in enumerate(reader):
        sample = (np.array([float(row[0]), float(row[1]), float(row[2]), float(row[3])]), float(row[4])) # 
        temp_loss = temp_loss + f.loss(sample)
        n = n + 1
    return temp_loss/n
    




