from __future__ import print_function
import timeit
#from keras.utils.generic_utils import Progbar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
import random

plt.ion()
from visualization import testGrid, plotKernelFunction

# ==================================================

class Callback(object):
    def __init__(self, config):
        pass
    def __getstate__(self):
        state = self.__dict__
        if 'env' in state: del state['env']
        if 'model' in state: del state['model']
        return state
    def set_params(self, params):
        self.params = params
    def set_model(self, model):
        self.model = model
    def set_env(self, env):
        self.env = env
    def on_test_begin(self, logs={}):
        pass
    def on_test_end(self, logs={}):
        pass
    def on_train_begin(self, logs={}):
        pass
    def on_train_end(self, logs={}):
        pass
    def on_episode_begin(self, episode, logs={}):
        pass
    def on_episode_end(self, episode, logs={}):
        pass
    def on_step_begin(self, step, logs={}):
        pass
    def on_step_end(self, step, logs={}):
        pass
    def on_action_begin(self, action, logs={}):
        pass
    def on_action_end(self, action, logs={}):
        pass

class CallbackList(object):
    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
    def append(self, callback):
        self.callbacks.append(callback)
    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)
    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)
    def set_env(self, env):
        for callback in self.callbacks:
            callback.set_env(env)
    def on_test_begin(self, logs={}):
        for callback in self.callbacks:
            callback.on_test_begin(logs)
    def on_test_end(self, logs={}):
        for callback in self.callbacks:
            callback.on_test_end(logs)
    def on_train_begin(self, logs={}):
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    def on_train_end(self, logs={}):
        for callback in self.callbacks:
            callback.on_train_end(logs)
    def on_episode_begin(self, episode, logs={}):
        for callback in self.callbacks:
            callback.on_episode_begin(episode, logs=logs)
    def on_episode_end(self, episode, logs={}):
        for callback in self.callbacks:
            callback.on_episode_end(episode, logs=logs)
    def on_step_begin(self, step, logs={}):
        for callback in self.callbacks:
            callback.on_step_begin(step, logs=logs)
    def on_step_end(self, step, logs={}):
        for callback in self.callbacks:
            callback.on_step_end(step, logs=logs)
    def on_action_begin(self, action, logs={}):
        for callback in self.callbacks:
            callback.on_action_begin(action, logs=logs)
    def on_action_end(self, action, logs={}):
        for callback in self.callbacks:
            callback.on_action_end(action, logs=logs)

# ==================================================

# Compute running averages for intervals of metrics
class IntervalMetrics(Callback):
    def __init__(self, config):
        self.step = 0
        self.interval = config.getint('ReportInterval', 10000)
        self.all_metrics = {}
        self.interval_metrics = {'step': []}
    def update(self, logs):
        self.interval_metrics['step'].append(self.step)
        for name, values in self.all_metrics.items():
            self.interval_metrics[name].append(np.nanmean(values[-self.interval:]))
        logs['interval_metrics'] = self.interval_metrics
        logs['all_metrics'] = self.all_metrics
    def on_train_begin(self, logs):
        self.metrics_names = self.model.metrics_names
        for name in self.metrics_names:
            self.all_metrics.setdefault(name, [])
            self.interval_metrics.setdefault(name, [])
    def on_train_end(self, logs):
        self.update(logs)
    def on_step_end(self, step, logs):
        for name, value in zip(self.metrics_names, logs['metrics']):
            self.all_metrics[name].append(value)
        self.step += 1
        if self.step % self.interval == 0:
            self.update(logs)

# ==================================================

# Collect episode rewards
class CollectRewards(Callback):
    def __init__(self, config):
        self.step = 0
        self.rewards = {'episode': [], 'step': [], 'reward': []}
    def update(self, logs):
        logs['all_episode_rewards'] = self.rewards
    def on_train_end(self, logs):
        self.update(logs)
    def on_episode_end(self, episode, logs):
        self.rewards['episode'].append(episode)
        self.rewards['step'].append(self.step)
        self.rewards['reward'].append(logs['episode_reward'])
        self.update(logs)
    def on_step_end(self, step, logs):
        self.step += 1
        self.update(logs)

# Collect episode rewards in intervals
class IntervalRewards(Callback):
    def __init__(self, config):
        self.step = 0
        self.interval = config.getint('ReportInterval', 10000)
    def reset(self):
        self.interval_rewards = []
    def update(self, logs):
        # Compute mean, min, and max of episode rewards
        logs['interval_rewards'] = {'nb_episodes': len(self.interval_rewards)}
        if len(self.interval_rewards) > 0:
            logs['interval_rewards']['mean'] = np.mean(self.interval_rewards)
            logs['interval_rewards']['min']  = np.min(self.interval_rewards)
            logs['interval_rewards']['max']  = np.max(self.interval_rewards)
    def on_train_begin(self, logs):
        self.reset()
    def on_train_end(self, logs):
        self.update(logs)
    def on_episode_end(self, episode, logs):
        self.interval_rewards.append(logs['episode_reward'])
    def on_step_end(self, step, logs):
        self.step += 1
        if self.step % self.interval == 0:
            self.update(logs)
            self.reset()

# ==================================================

class IntervalProgress(Callback):
    def __init__(self, config):
        self.step = 0
        self.interval = config.getint('ReportInterval', 10000)
    def __getstate__(self):
        state = super(IntervalProgress, self).__getstate__()
        #if 'progbar' in state: del state['progbar']
        return state
    def reset(self):

        self.interval_start = timeit.default_timer()
        #self.progbar = Progbar(target=self.interval)
        print('Interval {} ({} steps performed)'.format(self.step // self.interval + 1, self.step))
    def update(self, logs):
        formatted_metrics = ''
        if 'interval_metrics' in logs:
            for name, values in logs['interval_metrics'].items():
                if name == 'step': continue
                formatted_metrics += ' - {}: {:.4f}'.format(name, values[-1])
        formatted_rewards = ''
        if 'interval_rewards' in logs:
            eps = logs['interval_rewards']['nb_episodes']
            if eps > 0:
                formatted_rewards = ' - episode_rewards: {:.3f} [{:.3f}, {:.3f}]'.format(
                        logs['interval_rewards']['mean'],
                        logs['interval_rewards']['min'],
                        logs['interval_rewards']['max'])
        else:
            eps = 0
        print('{} episodes{}{}'.format(eps, formatted_rewards, formatted_metrics))
    def on_train_begin(self, logs):
        self.train_start = timeit.default_timer()
        print('Training for {} steps ...'.format(self.params['nb_steps']))
        self.reset()
    def on_train_end(self, logs):
        duration = timeit.default_timer() - self.train_start
        print('done, took {:.3f} seconds'.format(duration))
        self.update(logs)
    def on_step_end(self, step, logs):
        values = [('reward', logs['reward'])]
        #self.progbar.update((self.step % self.interval) + 1, values=values, force=True)
        self.step += 1
        if self.step % self.interval == 0:
            self.update(logs)
            self.reset()

# ==================================================

class IntervalTest(Callback):
    def __init__(self, config):
        self.step = 0
        self.interval   = config.getint('ReportInterval', 10000)
        self.testcount  = config.getint('TestCount', 0)
        self.testlength = config.getint('TestLength', 200)
    def set_env(self, env):
        # Make a new copy of the environment
        self.env = env.clone()
    def test(self, logs):
        if self.testcount <= 1: return
        total = 0.
        for _ in range(self.testcount):
            err = 0.
            s = self.env.reset()
            for i in range(self.testlength):
                a = self.model.act(s, stochastic=False)
                s_, r, done, _ = self.env.step(a)
                if done: s_ = None
                err += 0.5*self.model.bellman_error(s,a,r,s_)**2
                s = s_
                if done: break
            total += err / float(i+1)
        loss = float(total) / float(self.testcount) + self.model.model_error()
        logs.setdefault('interval_metrics',{}).setdefault('Testing Loss',[]).append(loss)
    def on_step_end(self, step, logs):
        self.step += 1
        if self.step % self.interval == 0:
            self.test(logs)

class IntervalACCTest(Callback):
    def __init__(self, config):
        self.step = 0
        self.interval = config.getint('ReportInterval', 10000)
        self.testpoints = config.getint('TestPoints', 2000)
    def set_env(self, env):
        # sample from environment state/actions
        self.samples = []
        for _ in range(self.testpoints):
            s, a = env.env.observation_space.sample(), env.env.action_space.sample()
            env.env.state = s
            s_, r, _, _ = env.env.step(a)
            self.samples.append((s, a, r, s_))
    def test(self, logs):
        err = 0.
        for s,a,r,s_ in self.samples:
            err += 0.5*self.model.bellman_error(s,a,r,s_)**2
        loss = float(err) / float(len(self.samples))
        logs.setdefault('interval_metrics',{}).setdefault('Testing Loss',[]).append(loss)
        logs.setdefault('interval_metrics',{}).setdefault('Regularized Testing Loss',[]).append(self.model.model.Q.normsq())
    def on_step_end(self, step, logs):
        self.step += 1
        if self.step % self.interval == 0:
            self.test(logs)

class IntervalMCTest(Callback):
    def __init__(self, config):
        self.x = []
        self.step = 0
        self.interval = config.getint('ReportInterval', 10000)

        self.teststatecount = config.getint('TestStateCount', 100) # count of test states
        self.testtrajlength = config.getint('TestTrajLength', 1000) # count of trajectory length

        #self.sarsa_steps = config.getint('SARSASteps', 100000)

    def set_env(self, env):
        # Make a new copy of the environment
        self.env = env.clone()

    def test(self, logs):
        #print(np.shape(self.x))
        if (not np.asarray(self.x).size == 0):
            perror = np.mean(np.abs(self.model.model.Q(self.x).flatten() - self.testValues) / np.abs(self.testValues))
        else:
            perror = 0
        logs.setdefault('interval_metrics', {}).setdefault('Testing Loss', []).append(perror)

    # Function to make one rollout
    def rollout(self,N=None, s=None):
        if s is None:
            s = self.env.reset()
        else:
            self.env.state = s
        tr = [] 
        while (N is None) or (len(tr) < N):

            if self.step == 1:
                stoch = True
            else:
                stoch = False

            a = self.model.act(s, stochastic=stoch) #policy.select(s)
            #print (a)
            s_, r, done, _ = self.env.step(a)
            if done:
                tr.append((s, a, r, None, None))
                return tr
            a_ = self.model.act(s_, stochastic=stoch) #policy.select(s_)
            tr.append((s, a, r, s_, a_))
            s = s_
        return tr

    # Function to make a trajectory (that could have multiple episodes)
    def make_trajectory(self,N):
        traj = []
        while len(traj) < N:
            #print (len(traj))
            traj.extend(self.rollout(N - len(traj)))
        return traj

    def mc_rollout(self):
        # Generate test trajectory
        testTrajectory = self.make_trajectory(self.testtrajlength)
        # Select test points
        samples = random.sample(testTrajectory, self.teststatecount)
        self.testStates = [tup[0] for tup in samples]
        self.testActions = [tup[1] for tup in samples]

        self.testActions = np.reshape(self.testActions,(-1,1))

        self.x = np.concatenate((self.testStates,self.testActions),axis=1)
        # Evaluate the rollouts from the test states
        self.testValues = []
        for i, s0 in enumerate(self.testStates):
            #print(i)
            # Perform many rollouts from each test state to get average returns
            R0 = 0.
            for k in range(self.testtrajlength):
                # Get the list of rewards
                Rs = [tup[2] for tup in self.rollout(2000, s0)]
                # Accumulate
                R = reduce(lambda R_, R: R + self.model.gamma * R_, Rs, 0.)
                # Average
                R0 += (R - R0) / (k + 1)
            # Save this value
            self.testValues.append(R0)
            #if (i + 1) % 100 == 0:
            #    print('Computing test point {}/{}'.format(i + 1, self.teststatecount))



    def on_step_end(self, step, logs):
        self.step += 1
        if self.step % self.interval == 1:
            self.test(logs)
        if self.step % self.sarsa_steps == 1:
            self.mc_rollout()

# ==================================================

class PlotMetrics(Callback):
    def __init__(self, config):
        self.prefix = config.name + ' - ' if config.name != 'DEFAULT' else ''
    def on_step_end(self, step, logs):
        if ('interval_metrics' in logs) and (len(logs['interval_metrics']['step']) > 1):
            for name, values in logs['interval_metrics'].items():
                if name == 'step': continue
                plt.figure(self.prefix + name); plt.clf()
                plt.plot(logs['interval_metrics']['step'], values)
                plt.title(self.prefix + name); plt.xlabel('Steps'); plt.ylabel(name)
            plt.draw_all(); plt.pause(1e-3)

class PlotRewards(Callback):
    def __init__(self, config):
        self.step = 0
        self.interval = config.getint('ReportInterval', 10000)
        self.prefix = config.name + ' - ' if config.name != 'DEFAULT' else ''
    def update(self, logs):
        if ('all_episode_rewards' in logs) and (len(logs['all_episode_rewards']['reward']) > 1):
            plt.figure(self.prefix + 'Cumulative Rewards'); plt.clf()
            plt.plot(logs['all_episode_rewards']['episode'], logs['all_episode_rewards']['reward'])
            plt.xlabel('Episode'); plt.ylabel('Cumulative Reward')
            plt.draw_all(); plt.pause(1e-3)
    def on_step_end(self, step, logs):
        self.step += 1
        if self.step % self.interval == 0:
            self.update(logs)

class PlotValueFunction(Callback):
    def __init__(self, config):
        self.step = 0
        self.interval = config.getint('ReportInterval', 10000)
        self.prefix = config.name + ' - ' if config.name != 'DEFAULT' else ''
    def update(self, logs):
        pass
    def on_train_begin(self, logs):
        self.bounds = self.env.bounds
    def on_step_end(self, logs):
        step += 1
        if self.step % self.interval == 0:
            self.update(logs)

# ==================================================
def make_callbacks(config):
    callbacks = CallbackList()
    for cls in Callback.__subclasses__():
        if config.getboolean(cls.__name__, False):
            callbacks.append(cls(config))
    return callbacks

