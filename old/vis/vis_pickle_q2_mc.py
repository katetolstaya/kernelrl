
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gym
from gym_gazebo.envs import gazebo_env
import random
#import pksarsa
#import kpolicy
#from function import KernelRepresentation

import imageio
images = []
folder = 'exp7_6_2'
num_files = 10

teststatecount = 100 # count of misc states
testtrajlength = 10000 # count of trajectory length
env = gym.make('PlanarAccel1-v0')
gamma = 0.99

def test(model, x, testValues): #(model,x,testValues):
    if (not np.asarray(x).size == 0): 
        perror = np.mean(np.abs(model(x).flatten() - testValues) / np.abs(testValues))
    else:
        perror = 0
    return perror


    # Function to make one rollout
def rollout(model, N=None, s=None):
    if s is None:
        s = env.reset()
    else: 
	env.state = s
    tr = [] 

    while (N is None) or (len(tr) < N):
        a = model.argmax(s)
        s_, r, done, _ = env.step(a)
        if done:
            tr.append((s, a, r, None, None))
            return tr
        a_ = model.argmax(s_)
        tr.append((s, a, r, s_, a_))
        s = s_
    return tr

    # Function to make a trajectory (that could have multiple episodes)
def make_trajectory(model, N):
    traj = []
    while len(traj) < N:
        traj.extend(rollout(model, N - len(traj)))
    return traj

def mc_rollout(model):

    # Generate misc trajectory
    testTrajectory = make_trajectory(model, testtrajlength)
    # Select misc points
    samples = random.sample(testTrajectory, teststatecount)
    testStates = [tup[0] for tup in samples]
    testActions = [tup[1] for tup in samples]

    testActions = np.reshape(testActions,(-1,1))

    x = np.concatenate((testStates,testActions),axis=1)
    # Evaluate the rollouts from the misc states
    testValues = []
    for i, s0 in enumerate(testStates):

        # Perform many rollouts from each misc state to get average returns
        R0 = 0.
        for k in range(testtrajlength):
            # Get the list of rewards
            Rs = [tup[2] for tup in rollout(model,2000, s0)]
            # Accumulate
            R = reduce(lambda R_, R: R + gamma * R_, Rs, 0.)
            # Average
            R0 += (R - R0) / (k + 1)
            # Save this value
        testValues.append(R0)

    return (x, testValues)

error = np.zeros((num_files,1))

for n in xrange(1,num_files):
	method = 'DEFAULT'
	f =  open('../rlcore/'+folder+'/kpolicy_model_'+str(n)+'.pkl','rb')
	model = cPickle.load(f)
	(x, testValues) = mc_rollout(model)
        error[n] = test(model, x, testValues)
	#print error[n]
        

plt.plot(np.linspace(0, num_files, num_files), error)
plt.title('MC Test Loss')
plt.xlabel('Training Epoch')
plt.ylabel('Test Loss')
plt.show()
