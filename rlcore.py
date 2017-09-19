from __future__ import print_function
import numpy as np, gym
from backports import configparser
import pickle
from callbacks import CallbackList, make_callbacks
import sys
sys.path.append('../gym_gazebo/envs')
# Our agent types

from kv import KVAgent
from kqlearning import KQLearningAgent
from ksarsa import KSARSAAgent
from kqlearning2 import KQLearningAgent2
from policy_test import QTestAgent2
from kpolicy import KPolicyAgent
from kpolicytab import KPolicyTabAgent
from kdpg import KDPGAgent
from kqtab import KQTabAgent
from tqdm import tqdm

import copy

import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

import matplotlib
import matplotlib.pyplot as plt
import sys
matplotlib.use('Agg')
# ==================================================

class Environment:
    def __init__(self, cfg):
        problem  = cfg.get('GymEnvironment')
	self.reset_state = cfg.getboolean('ResetState',False)
	print(problem)
        self.env = gym.make(problem)
	#print (self.env.action_space.high)
	#print (self.env.action_space.low)
    @property
    def stateCount(self):
        return self.env.observation_space.shape[0]
    @property
    def actionCount(self):
	if hasattr(self.env.action_space, 'n'):
        	return self.env.action_space.n
	else:
		return self.env.action_space.shape[0]
    @property
    def bounds(self):
        return np.vstack((self.env.observation_space.high, self.env.observation_space.low))
    def clone(self):
        return self.env.spec.make()

    # ----------------------------------------
    # Run one step of Q-learning training
    # ----------------------------------------
    def run(self, agent, nb_steps=None, episode=-1, callbacks=CallbackList()):
        # Total cumulative reward
        R = 0.
        # Episode steps
        episodeStep = 0
        # Reset state
        s = self.env.reset()

	#if nb_steps is not None:
	#    pbar = tqdm(total=nb_steps)

        while (nb_steps is None) or (episodeStep < nb_steps):
	    if self.reset_state:
	    	s = self.env.reset()



            callbacks.on_step_begin(episodeStep)

            # Choose action
            a = agent.act(s)


            # Take action
            callbacks.on_action_begin(a)
            #self.env.render()
            s_, r, done, info = self.env.step(a)
	    
	    #print (s_,a)
	    #print (s)
            callbacks.on_action_end(a)
            # Process this transition

            agent.observe( (s, a, r, None if done else s_) )
            metrics = agent.improve()

            # Prepare for next step
            callbacks.on_step_end(episodeStep, {
                'action': a,
                'observation': s_,
                'reward': r,
                'metrics': metrics,
                'episode': episode,
                'info': info,
                })
            R += r
            episodeStep += 1

	    #if nb_steps is not None:
	    #    pbar.update(1)
            
            #if episodeStep >= self.reset_state and self.reset_state > 0:
	    #   done = True

            s = copy.deepcopy(s_)
            if done:
                break

	#if nb_steps is not None:
	#    pbar.close()

        # End of episode
	sys.stdout.flush()
        return R, episodeStep

    # ----------------------------------------
    # Run one step of SARSA training
    # ----------------------------------------
    def runSARSA(self, agent, nb_steps=None, episode=-1, callbacks=CallbackList()):
        # Total cumulative reward
        R = 0.
        # Episode steps
        episodeStep = 0
        # Reset state
        s = self.env.reset()
        # Choose action
        a = agent.act(s)

        while (nb_steps is None) or (episodeStep < nb_steps):
	    if self.reset_state:
	    	s = self.env.reset()
            callbacks.on_step_begin(episodeStep)

            # Take action
            callbacks.on_action_begin(a)
            #self.env.render()

            s_, r, done, info = self.env.step(a)
            callbacks.on_action_end(a)
            # Choose next action
            a_ = agent.act(s_)

 
		
            # Process this transition
            agent.observe( (s, a, r, None if done else s_, None if done else a_) )

            metrics = agent.improve()

            # Prepare for next step
            callbacks.on_step_end(episodeStep, {
                'action': a,
                'observation': s_,
                'reward': r,
                'metrics': metrics,
                'episode': episode,
                'info': info,
                })
            R += r
            episodeStep += 1
            s = copy.deepcopy(s_)
            a = copy.deepcopy(a_)

            if done:
                break

        # End of episode
        return R, episodeStep

    # ----------------------------------------
    # Visualize an agent policy
    # ----------------------------------------
    def visualize(self, agent, nb_steps=None):
        "Visualize the agent for some number of steps."
        s = self.env.reset()
        step = 0
        while (nb_steps is None) or (step < nb_steps):
            self.env.render()
            a = agent.act(s, stochastic=False)
            s, _, done, _ = self.env.step(a)
            step += 1
            if done: break

# ==================================================

class Experiment(object):
    def __init__(self, config):
        # Create our environment
        # ------------------------------
        self.env = Environment(config)

        # Create our agent
        # ------------------------------
        atype = config.get('Agent')
        if atype.lower() == 'kv':
            self.agent = KVAgent(self.env, config)
        elif atype.lower() == 'kqlearning':
            self.agent = KQLearningAgent(self.env, config)
        elif atype.lower() == 'kqlearning2':
                self.agent = KQLearningAgent2(self.env, config)
        elif atype.lower() == 'qtest':
                self.agent = QTestAgent2(self.env, config)
        elif atype.lower() == 'kpolicy':
                self.agent = KPolicyAgent(self.env, config)
        elif atype.lower() == 'kpolicytab':
                self.agent = KPolicyTabAgent(self.env, config)
        elif atype.lower() == 'kqtab':
                self.agent = KQTabAgent(self.env, config)
        elif atype.lower() == 'ksarsa':
            self.agent = KSARSAAgent(self.env, config)
        elif atype.lower() == 'kdpg':
            self.agent = KDPGAgent(self.env, config)
        elif atype is None:
            raise ValueError("'Agent' type not specified")
        else:
            raise ValueError("Invalid 'Agent' type: {}".format(atype))

        # Start progress
        # ------------------------------
        self.episode = 0
        self.steps   = 0
        # Number of steps
        if 'TrainingSteps' not in config:
            raise ValueError("'TrainingSteps' not specified")
        self.maximum_steps = config.getint('TrainingSteps')

        # Load reporting callbacks
        # ------------------------------
        self.callbacks = make_callbacks(config)
        self.callbacks.set_env(self.env)
        self.callbacks.set_model(self.agent)
        self.callbacks.set_params({'nb_steps': self.maximum_steps})

    def run(self):
        # Begin our training
        self.callbacks.on_train_begin()
        # Select how our environment runs an episode
        if isinstance(self.agent, KSARSAAgent) or isinstance(self.agent, KPolicyAgent) or isinstance(self.agent, KPolicyTabAgent) or isinstance(self.agent, KDPGAgent):
            erun = self.env.runSARSA
        else:
            erun = self.env.run

        # Run until we are out of training steps
        while self.steps < self.maximum_steps:
            # Beginning of an episode
            self.callbacks.on_episode_begin(self.episode)
            R, episodeSteps = erun(self.agent,
                    nb_steps = self.maximum_steps - self.steps,
                    episode = self.episode,
                    callbacks = self.callbacks)
            self.steps += episodeSteps
            self.episode += 1
            self.callbacks.on_episode_end(self.episode, {
                'episode_reward': R,
                'nb_episode_steps': episodeSteps,
                'nb_steps': self.steps
                })
	    print(episodeSteps)


        # Collect final information
        logs = {}
        logs['agent'] = self.agent
	#print (self.agent.model.Q.KDD)
        self.callbacks.on_train_end(logs)
        return logs

# ==================================================

# ------------------------------
# Run a batch of experiments
# ------------------------------
def run_experiments(config):
    print(config)
    if isinstance(config, str):
        fname = config
        config = configparser.ConfigParser()
        with open(fname, 'r') as f:
            config.read_file(f)

    # The output of our experiments
    experiments = {}
    # Check if there are multiple experiments to run
    if config.sections():
        for sectionName in config.sections():
            # Create the experiment
            experiment = Experiment(config[sectionName])
            # Run and append to our output
            experiments[sectionName] = experiment.run()
    else:
        # Create our default experiment
        experiment = Experiment(config[config.default_section])
     # Run and append to our output
        experiments[config.default_section] = experiment.run()
    # Return data
    return experiments

if __name__ == '__main__':
    #run_experiments(sys.argv[1])
    #run_experiments('ksarsa.cfg')
    #run_experiments('kq.cfg')
    #cfg/kpolicy_quad.cfg
    #cfg/kq_quad2.cfg
    ret = run_experiments('cfg/kq_mccar2.cfg')
#kq_mccar_multi

#('cfg/kq_mccar_multi2.cfg')
#('cfg/kq_mccar_multi2.cfg')
#('cfg/peval.cfg')
#('cfg/kq_mccar_multi2.cfg')
#('cfg/peval.cfg')
#
#'cfg/kq_quadx.cfg')
#'cfg/kdpg_quad2.cfg') # 'cfg/kq_planar1.cfg') # #)
    #cProfile.run(run_experiments('cfg/kq_planar1.cfg'))
	#ret = run_experiments('cfg/peval_quad2.cfg')
    with open('exp_9_10.pkl', 'wb') as f:
        pickle.dump(ret, f)
    #run_experiments('kq.cfg')

    #wait = input("PRESS ENTER TO CONTINUE.")
