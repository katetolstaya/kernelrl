import copy, json, random
import numpy as np
from corerl.core import ScheduledParameter
from corerl.function import KernelRepresentation
import pdb
import pickle
import csv
import matplotlib.pyplot as plt





def model_error(f):
    return 0.5 * lossL * f.normsq()

def predict(f,x):  # Predict the Q function values for a batch of states.
    return f(x)

def predictOne(f,x):  # Predict the Q function values for a single state.
    return f(np.reshape(x,(1,-1)))[0]

    
def val(f,x):
    return predictOne(f,x)[0] + mu

def mom(f,x):
    return predictOne(f,x)[1]
    
def var(f,x):
    return predictOne(f,x)[2]  + sigma

def train(f, sample):

    x,y = sample
    grad = (val(f,x)-y)
    grad_sq = grad**2
     
    # V gradient
    W = np.zeros((3,))
    
    if mult == 0: # simple SGD
        W[0] =  -eta * grad
    elif mult == 1:
        grad_var = grad_sq
        W[0] =  -eta * grad / np.sqrt(abs(var(f,x))+0.1)
        W[2] = (-beta2 * var(f,x) + beta2 * grad_var) * mult
    elif mult == 2:
        grad_var = (grad-mom(f,x))**2
        W[0] =  -eta * mom(f,x) / np.sqrt(abs(var(f,x))+0.1) 
        W[1] = (-beta1 * mom(f,x) + beta1 * grad) * mult
        W[2] = (-beta2 * var(f,x) + beta2 * grad_var) * mult
    else:
        print('error')
    
    # Gradient step
    f.shrink(1. - lossL)
    
    #W[2] = -beta2 * var(f,x) + beta2 * 1/(grad_sq + 0.1)
    #W[0] =  -eta * (mom(f,x)+ W[1])/beta1 * np.sqrt(np.abs(var(f,x)+W[2])/beta2)

    f.append(np.array(x), np.reshape(W, (1, -1)))
    # Prune
    f.prune(eps)
    
    return (grad_sq/2, len(f.D))
    
def loss(f, sample):
    x, y = sample
    return 0.5 * (val(f,x)-y)**2

    
def train_model(fname, mult=1):
    f = KernelRepresentation(4, 3, config)

    csvfile = open(fname)
    reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    next(reader, None)
    
    losses = []

    for i, row in enumerate(reader):
        sample = (np.array([float(row[0]), float(row[1]), float(row[2]), float(row[3])]), float(row[4]))
        (loss,model) =  train(f, sample, mult)
        losses.append(loss)
        
    #plt.plot(range(0, len(losses)), losses)
    #plt.show()
    #print('Mean loss out of last 100')
    #print(np.mean(losses[-100:]))
        
    return f
    
def test_model(f, fname):
    csvfile = open(fname)
    reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    next(reader, None)

    n = 0
    temp_loss = 0
    for i, row in enumerate(reader):
        sample = (np.array([float(row[0]), float(row[1]), float(row[2]), float(row[3])]), float(row[4])) # 
        temp_loss = temp_loss + loss(f, sample)
        n = n + 1
        
    return temp_loss/n
    
def point_density(f, x):
    return np.sum(f.kernel.f(x, f.D))
    

def compose(f1, f2, mult):
    f = KernelRepresentation(4, 3, config)
    d = np.vstack([f1.D, f2.D]) 
    
    thresh = np.shape(f1.D)[0]
    
    W = np.zeros((3,))
    for i in np.random.permutation(np.shape(d)[0]):
        x = d[i,:]
        if mult == 0:
            if point_density(f1, x) >= point_density(f2, x) and i < thresh: # reciprocal here so larger is better
                W[1] = -mom(f,x) + mom(f1,x)
                W[2] = -var(f,x) + var(f1,x)
                W[0] = -val(f,x) + val(f1,x)
                f.append(np.array(x), np.reshape(W, (1, -1)))
            elif point_density(f1, x) < point_density(f2, x) and i >= thresh:
                W[1] = -mom(f,x) + mom(f2,x)
                W[2] = -var(f,x) + var(f2,x)
                W[0] = -val(f,x) + val(f2,x)
                f.append(np.array(x), np.reshape(W, (1, -1)))
        elif mult == 1 or mult==2:
            if var(f1,x) <= var(f2,x) and i < thresh: # reciprocal here so larger is better
                W[1] = -mom(f,x) + mom(f1,x)
                W[2] = -var(f,x) + var(f1,x)
                W[0] = -val(f,x) + val(f1,x)
                f.append(np.array(x), np.reshape(W, (1, -1)))
            elif var(f1,x) > var(f2,x) and i >= thresh:
                W[1] = -mom(f,x) + mom(f2,x)
                W[2] = -var(f,x) + var(f2,x)
                W[0] = -val(f,x) + val(f2,x)
                f.append(np.array(x), np.reshape(W, (1, -1)))
    return f
    
def train_test_compose(mult):
    f1 = train_model('data/ccpp1.csv', mult)
    #print('Trained 1, dataset 1')
    test1_1 = test_model(f1, 'ccpp1.csv')
    test1_2 = test_model(f1, 'ccpp2.csv')
    print('Trained with D1, Tested with D1: ' + str(test1_1) + ', D2: ' + str(test1_2) )

    f2 = train_model('data/ccpp2.csv', mult)
    test2_1 = test_model(f2, 'ccpp1.csv')
    test2_2 = test_model(f2, 'ccpp2.csv')
    print('Trained with D2, Tested with D1: ' + str(test2_1) + ', D2: ' + str(test2_2) )

    f = compose(f1,f2, mult)
    test = test_model(f, 'data/ccpp.csv')
    print('Composed, Tested with D: ' + str(test) )

def train_test(mult):
    f0 = train_model('data/ccpp.csv', mult)
    test0 = test_model(f0, 'ccpp.csv')
    print('Trained with D, tested with D: '+ str(test0))
    
    
print('Train with SGD')
train_test(0)
print('Train with raw 2nd moment SGD')
train_test(1)
print('Train with momentum and variance')
train_test(2)

print('Compose with SGD')
train_test_compose(0)
print('Compose with raw 2nd moment SGD')
train_test_compose(1)
print('Compose with momentum and variance')
train_test_compose(2)



