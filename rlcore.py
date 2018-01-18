from __future__ import print_function

import copy
import logging
import pickle
import sys

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser



import gym
import numpy as np

logger = logging.getLogger(__name__)

# Our agent types
from corerl.callbacks import CallbackList, make_callbacks
from algs.kqlearning import KQLearningAgent
from algs.knaf import KNAFAgent
from misc.knaf2 import KNAF2Agent
from algs.ksarsa import KSARSAAgent
from algs.kqlearning2 import KQLearningAgent2
from algs.policy_test import QTestAgent2
from algs.kpolicy import KPolicyAgent
from algs.kpolicytab import KPolicyTabAgent
from algs.kdpg import KDPGAgent
from algs.kqtab import KQTabAgent
from algs.kadv import KAdvAgent
from algs.kqgreedy import KGreedyQAgent
from algs.kqgreedy_per import KQGreedyAgentPER
from algs.kqlearning_per import KQLearningAgentPER
from algs.kqlearning_iid import KQLearningAgentIID
from corerl.random_agent import RandomAgent
from algs.kqlearningalt import KQLearningAltAgent


# from kv import KVAgent

# sys.path.append('../gym_gazebo/envs')
# from tqdm import tqdm
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Agg')
# ==================================================

class Environment:
    def __init__(self, cfg):
        problem = cfg.get('GymEnvironment')
        self.reset_state = cfg.getboolean('ResetState', False)
        print(problem)
        self.env = gym.make(problem)

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

        while (nb_steps is None) or (episodeStep < nb_steps):
            if self.reset_state:
                s = self.env.reset()

            callbacks.on_step_begin(episodeStep)
            # Choose action
            a = agent.act(s)

            # Take action
            callbacks.on_action_begin(a)
            # self.env.render()
            s_, r, done, info = self.env.step(a)
            callbacks.on_action_end(a)
            # Process this transition

            agent.observe((s, a, r, None if done else s_))
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
            if done:
                break

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
            # self.env.render()

            s_, r, done, info = self.env.step(a)
            callbacks.on_action_end(a)
            # Choose next action
            a_ = agent.act(s_)

            # Process this transition
            agent.observe((s, a, r, None if done else s_, None if done else a_))

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
        if atype.lower() == 'kqlearning':
            self.agent = KQLearningAgent(self.env, config)
        elif atype.lower() == 'kqlearningper':
            self.random_agent = RandomAgent(self.env, config)
            self.agent = KQLearningAgentPER(self.env, config)
        elif atype.lower() == 'kqgreedyper':
            self.random_agent = RandomAgent(self.env, config)
            self.agent = KQGreedyAgentPER(self.env, config)
        elif atype.lower() == 'kqlearningiid':
            self.random_agent = RandomAgent(self.env, config)
            self.agent = KQLearningAgentIID(self.env, config)
        elif atype.lower() == 'kqlearning2':
            self.agent = KQLearningAgent2(self.env, config)
        elif atype.lower() == 'kqlearningalt':
            self.agent = KQLearningAltAgent(self.env, config)
        elif atype.lower() == 'kadv':
            self.agent = KAdvAgent(self.env, config)
        elif atype.lower() == 'knaf':
            self.agent = KNAFAgent(self.env, config)
        elif atype.lower() == 'knaf2':
            self.agent = KNAF2Agent(self.env, config)
        elif atype.lower() == 'kgreedyq':
            self.agent = KGreedyQAgent(self.env, config)
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
        self.steps = 0
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
        # Initialize our memory
        if hasattr(self.agent, 'memory'):
            print('Initializing memory with random agent...')
            while not self.random_agent.memory.isFull():
                self.env.run(self.random_agent, nb_steps=self.random_agent.memory.remaining())
                print('%d/%d' % (self.random_agent.memory.length, self.random_agent.memory.capacity))
            self.agent.memory = self.random_agent.memory
            del self.random_agent

        # Begin our training
        self.callbacks.on_train_begin()
        # Select how our environment runs an episode
        if isinstance(self.agent, KSARSAAgent) or isinstance(self.agent, KPolicyAgent) or isinstance(self.agent,
                                                                                                     KPolicyTabAgent) or isinstance(
                self.agent, KDPGAgent):
            erun = self.env.runSARSA
        else:
            erun = self.env.run

        # Run until we are out of training steps
        while self.steps < self.maximum_steps:
            # Beginning of an episode
            self.callbacks.on_episode_begin(self.episode)
            R, episodeSteps = erun(self.agent,
                                   nb_steps=self.maximum_steps - self.steps,
                                   episode=self.episode,
                                   callbacks=self.callbacks)
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
        self.callbacks.on_train_end(logs)
        return logs


# ==================================================
# ------------------------------
# Run a batch of experiments
# ------------------------------
def run_experiments(config):
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

    # Some options for CFG files:

    # 'ksarsa.cfg', 'kq.cfg', kpolicy_quad.cfg, cfg/kq_quad2.cfg, 'cfg/kq_mccar_multi2.cfg'
    # 'cfg/knaf_pendulum.cfg', 'cfg/kq_robot.cfg', 'cfg/kq_pendulum.cfg', 'cfg/kadv_pendulum.cfg', 'cfg/knaf_mcar.cfg',
    # 'cfg/kq_pendulum.cfg', 'cfg/knaf_pendulum_multi.cfg', 'cfg/kq_pendulum_iid.cfg', 'cfg/kq_invpendulum.cfg'

    # fname = sys.argv[1]

    #fname = 'cfg/kgreedyq_pendulum.cfg'
    fname = 'cfg/kqalt_pendulum.cfg'
    #fname = 'cfg/kqgreedy_pendulum_per.cfg'

    print(fname)
    if isinstance(fname, str):
        config = ConfigParser()
        with open(fname, 'r') as f:
            config.read_file(f)

        ret = run_experiments(config)
        # cProfile.run(run_experiments('cfg/kq_planar1.cfg'))

        if 'PKLFileName' in config:
            pkl_fname = config.get('PKLFileName')
            with open(pkl_fname, 'wb') as f:
                pickle.dump(ret, f)
