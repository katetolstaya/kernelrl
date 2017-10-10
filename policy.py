import numpy as np
import numpy.random as nr
import h5py
import cPickle
import sys
#from VelocityController import VelocityController
#from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
sys.path.append('../modular_rl')

# ------------------------------
# Discrete Control Policy
# ------------------------------
# Probability of choosing action i is effectively the softmax score
#  p(s, a) = exp(w_a' * s) / sum_a exp(w_a' * s)
# class VelocityControlPolicy:
#     # n = size of state space
#     # m = size of action space
#     def __init__(self, config):
#         # self.W = nr.randn(n, m)
#         # self.actions = range(m)

#         target = Pose()
#         target.position.x = 0
#         target.position.y = 0
#         target.position.z = 0

#         self.vController = VelocityController()
#         self.vController.setTarget(target)
#     def select(self, state):
#         pos = PoseStamped()
#         pos.pose.position.x = state[0]
#         pos.pose.position.y = state[1]
#         pos.pose.position.z = state[2]
#         pos.pose.orientation.x = state[3]
#         pos.pose.orientation.y = state[4]
#         pos.pose.orientation.z = state[5]
#         pos.pose.orientation.w = state[6]

#         act = self.vController.update(pos)
#         action = np.zeros(6,)

#         action[0] = act.twist.linear.x
#         action[1] = act.twist.linear.y
#         action[2] = act.twist.linear.z
#         action[3] = act.twist.angular.x
#         action[4] = act.twist.angular.y
#         action[5] = act.twist.angular.z

#         # exps = np.exp(s.dot(self.W))
#         # return nr.choice(self.actions, p=exps/exps.sum())
#         return action
#     def reset(self):
#         pass

# ------------------------------
# Discrete Control Policy
# ------------------------------
# Probability of choosing action i is effectively the softmax score
#  p(s, a) = exp(w_a' * s) / sum_a exp(w_a' * s)
# class Velocity2ControlPolicy:
#     # n = size of state space
#     # m = size of action space
#     def __init__(self, config):
#         # self.W = nr.randn(n, m)
#         # self.actions = range(m)

#         target = Pose()
#         target.position.x = 0
#         target.position.y = 0
#         target.position.z = 0

#         self.vController = VelocityController()
#         self.vController.setTarget(target)
#     def select(self, state):
#         pos = PoseStamped()
#         pos.pose.position.x = state[0]
#         pos.pose.position.y = state[1]


#         act = self.vController.update(pos)
#         action = np.zeros(2,)

#         action[0] = act.twist.linear.x
#         action[1] = act.twist.linear.y

#         # exps = np.exp(s.dot(self.W))
#         # return nr.choice(self.actions, p=exps/exps.sum())
#         return action
#     def reset(self):
#         pass

# ------------------------------
# Discrete Control Policy
# ------------------------------
# Probability of choosing action i is effectively the softmax score
#  p(s, a) = exp(w_a' * s) / sum_a exp(w_a' * s)
class RandomControlPolicy:
    # n = size of state space
    # m = size of action space
    def __init__(self, config):
	self.max_act = 5
	self.actionCount = 2
    def select(self, state):
	return np.random.uniform(-self.max_act,self.max_act,(self.actionCount,1))
    def reset(self):
        pass

# ------------------------------
# Discrete Control Policy
# ------------------------------
# Probability of choosing action i is effectively the softmax score
#  p(s, a) = exp(w_a' * s) / sum_a exp(w_a' * s)
class DiscreteControlPolicy:
    # n = size of state space
    # m = size of action space
    def __init__(self, n, m):
        self.W = nr.randn(n, m)
        self.actions = range(m)

    def select(self, s):
        exps = np.exp(s.dot(self.W))
        return nr.choice(self.actions, p=exps / exps.sum())

    def reset(self):
        pass

# ------------------------------
# 'module_rl' Control Policy
# ------------------------------
class TRPOSnapshotPolicy:
    def __init__(self, config):
        self.fname  = config.get('PolicyFile')
        self.number = config.getint('PolicySnapshotNumber')
        hdf = h5py.File(self.fname,'r')
        #TODO
        print(self.fname)
        self.agent = cPickle.loads(hdf['agent_snapshots']['%04d'%self.number].value)
        self.agent.stochastic = False
    def select(self, s):
        ob = self.agent.obfilt(s)
        a, _info = self.agent.act(ob)
        return a
    def reset(self):
        if hasattr(self.agent, 'reset'):
            self.agent.reset()
    def matches(self, config):
        return (config.get('PolicyType').lower() == 'trposnapshot') and (config.get('PolicyFile') == self.fname) and (config.getint('PolicySnapshotNumber') == self.number)


class QPolicy:
    def __init__(self, config):
        self.fname  = config.get('PolicyFile')
	self.model = cPickle.load(open(self.fname,'rb'))

        #self.number = config.getint('PolicySnapshotNumber')
        #hdf = h5py.File(self.fname,'r')
        #TODO
        print(self.fname)
        #self.agent = cPickle.loads(hdf['agent_snapshots']['%04d'%self.number].value)
        #self.agent.stochastic = False
    def select(self, s):
        return self.model.argmax(s)
    def reset(self):
	pass
    def matches(self, config):
        pass

# ==================================================
def make_policy(config):
    policyType = config.get('PolicyType').lower()
    if policyType == 'trposnapshot':
        return TRPOSnapshotPolicy(config)
    if policyType == 'velocity':
        return VelocityControlPolicy(config)
    if policyType == 'velocity2':
        return Velocity2ControlPolicy(config)
    if policyType == 'qpolicy':
        return QPolicy(config)
    else:
        raise ValueError('Unknown policy type: %s' % config.get('PolicyType'))

