import copy, json, random
import numpy as np
from corerl.core import ScheduledParameter
from corerl.function import KernelRepresentation
import pdb
import pickle
import csv

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser

# TODO: set up config file

# TODO: read in training data

# TODO: compute test error
fname = 'cfg/kadam.cfg'

config = ConfigParser()
with open(fname, 'r') as f:
    config.read_file(f)
    
config = config[config.default_section]

# Represent V, pi, L in one RKHS
f = KernelRepresentation(4, 3, config)

# Learning rates
eta = 0.1

beta = 0.1

# Regularization
lossL =  1e-6

# Representation error budget
eps = 0.1

sigma = 1000


with open('ccpp.csv') as csvfile:
    reader = csv.DictReader(csvfile)

    for i, row in enumerate(reader):
        sample = (np.array([float(row['AT']), float(row['V']), float(row['AP']), float(row['RH'])]), float(row['PE'])))
        err.append(train(f, sample))
        
        print (err[i])



def model_error(f):
    return 0.5 * lossL * f.normsq()

def predict(f,x):  # Predict the Q function values for a batch of states.
    return f(x)

def predictOne(f,x):  # Predict the Q function values for a single state.
    return f(np.reshape(x,(1,-1)))

    
def val(f,x):
    return predictOne(f,x)[0]

def mom(f,x):
    return predictOne(f,x)[1]
    
def var(f,x):
    return predictOne(f,x)[2] + sigma

def train(f, sample):

    x,y = sample
    grad = (val(x)-y)
    grad_sq = grad**2
    
    # Gradient step
    f.shrink(1. - self.lossL)

    # V gradient
    W = np.zeros((1,))

    W[1] = (beta-1) * mom(x) + (1-beta) * grad
    W[2] = (beta-1) * var(x) + (1-beta) * grad_sq
    W[0] =  - eta * mom(x)/np.sqrt(var(x))

    f.append(np.array(x), np.reshape(W, (1, -1)))
    # Prune
    f.prune(eps)
    return grad_sq



