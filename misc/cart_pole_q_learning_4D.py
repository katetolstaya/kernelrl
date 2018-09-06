import gym
import numpy as np
import random
import math
import sys
import copy
from tqdm import tqdm, trange
from gym.envs.registration import register
from time import sleep
sys.path.append('../gym_gazebo/envs')
from gazebo_env import GazeboEnv
from gym_gazebo.envs import gazebo_env
from gym_gazebo.envs import gazebo_px4copter_hover
from gym_gazebo.envs import gazebo_px4copter_vel


## Initialize the "Cart-Pole" environment
env = gym.make('PlanarAccel1-v0') #gym.make('CartPole-v0')
#env = gym.make('MountainCarContinuous-v0')

## Defining the environment related constants

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = (20,20) #(1, 1, 6, 3)  # (x, x', theta, theta')
# Number of discrete actions
NUM_ACTIONS = 5 #env.action_space.n # (left, right)
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

#STATE_BOUNDS[1] = [-0.5, 0.5]
#STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]
# Index of the action
ACTION_INDEX = len(NUM_BUCKETS)

## Creating a Q-Table for each state-action pair
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

## Learning related constants
MIN_EXPLORE_RATE = 0.1
MIN_LEARNING_RATE = 0.005

## Defining the simulation related constants
NUM_EPISODES =  100000
MAX_T = 499
STREAK_TO_END = 200
SOLVED_T = 0
DEBUG_MODE = True

def simulate():

    ## Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99  # since the world is unchanging

    num_streaks = 0

    tqdm_num_ep = trange(NUM_EPISODES)
    avg_len = 500

    for episode in tqdm_num_ep:
	tqdm_num_ep.set_description('Episode #%d' % episode)

        # Reset the environment
        obv = env.reset()

        # the initial state
        state_0 = state_to_bucket(obv)
	last_q = copy.deepcopy(q_table)

	if episode % 500 == 0:
	    print avg_len

        for t in range(MAX_T):
            env.render()

            # Select an action
            action = select_action(state_0, explore_rate)

            # Execute the action
	    action0 = np.rint((action+2))
            obv, reward, done, _ = env.step(action)

            # Observe the result
            state = state_to_bucket(obv)


            # Update the Q based on the result
            best_q = np.amax(q_table[tuple(state)])
	    
	    x = np.append(state_0 , int(action0))
	    #print (x)
	    #print(q_table[tuple(x)])
            q_table[tuple(x)] += learning_rate*(reward + discount_factor*(best_q) - q_table[tuple(x)])

            # Setting up for the next iteration
            state_0 = state #copy.deepcopy(state)

            # Print data
            #if (DEBUG_MODE):
            #    print("\nEpisode = %d" % episode)
            #    print("t = %d" % t)
            #    print("Action: %s" % str(action0))
            #    print("State: %s" % str(obv))
            #    print("Reward: %f" % reward)
            #    print("Best Q: %f" % best_q)
            #    print("Explore rate: %f" % explore_rate)
            #    print("Learning rate: %f" % learning_rate)
            #    #print("Streaks: %d" % num_streaks)

            #    print("")

            if done:
               #print("Episode %d finished after %f time steps" % (episode, t))
               # >= SOLVED_T):

               #num_streaks += 1
               break
	avg_len = avg_len*0.99 + t*0.01
	if not done:
	    num_streaks = 0

	tqdm_num_ep.set_postfix(loss=num_streaks) #np.linalg.norm(last_q-q_table))
            #sleep(0.25)

        # It's considered done when it's solved over 120 times consecutively
        #if num_streaks > STREAK_TO_END:
        #    break

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)

    print q_table
    print avg_len
    np.save('table_tab2_427', q_table)


def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = np.rint(env.action_space.sample())
    # Select the action with the highest q
    else:
	b = q_table[tuple(state)]
        action = np.array([np.random.choice(np.flatnonzero(b == b.max()))-2]) 
	#print action
	#
    #print action
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10(float(t+1)/NUM_EPISODES*10)))

def get_learning_rate(t):
    return MIN_LEARNING_RATE
    #return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10(float(t+1)/NUM_EPISODES*10)))

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*float(STATE_BOUNDS[i][0])/bound_width
            scaling = float(NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

if __name__ == "__main__":
    simulate()
