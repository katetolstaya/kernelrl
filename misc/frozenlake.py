import gym
import numpy as np
import scipy
import random

##########################################################
gradstep = False  # switch between gradient step and q learning
usebuffer = False  # use buffer or follow trajectories?


##########################################################

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x))


env = gym.make('FrozenLake-v0')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters
lr = 0.5  # 0.5
y = 0.95  # .95
num_episodes = 10000

# create lists to contain total rewards and steps per episode
rList = []

# buffer params
bufferlen = 1000
buffersamples = np.zeros((bufferlen, 4), dtype=int)
ind = 0
filled = False
eps = 0.1

for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Table learning algorithm
    while j < 99:
        j += 1
        # Epsilon greedy policy
        if random.random() >= eps:
            a = np.argmax(Q[s, :])
        else:
            a = np.random.randint(0, env.action_space.n)

        eps = 0.1 + 0.9 * float(num_episodes - i) / num_episodes  # (1./(0.001*i+1))
        lr = 0.1 + 0.4 * float(num_episodes - i) / num_episodes
        # lr = (1./(0.001*i+1)) # decreasing step size
        # Get new state and reward from environment
        s1, r, d, _ = env.step(a)
        rAll += r
        stemp = s1

        # save to buffer
        if usebuffer:
            buffersamples[ind, :] = [s, a, r, s1]
            ind = ind + 1
            if ind + 1 == bufferlen:
                ind = 0
                filled = True

        # train on sample from buffer
        if filled or not usebuffer:
            if usebuffer:
                s, a, r, s1 = buffersamples[int(np.random.uniform(0, bufferlen)), :]

                # Update Q-Table with new knowledge
            a1 = np.argmax(Q[s1, :])
            td = r + y * Q[s1, a1] - Q[s, a]
            # td = r + y*scipy.misc.logsumexp(Q[s1,:]) - Q[s,a]

            # gradient step
            if gradstep:
                Q[s1, a1] = Q[s1, a1] - lr * y * td  # comment this for q-learning
            Q[s, a] = Q[s, a] + lr * td

        s = stemp  # next state
        if d == True:
            break
    rList.append(rAll)

print ("Score over time: " + str(sum(rList) / num_episodes))

print ("Final Q-Table Values")
print (Q)
