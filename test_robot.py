import gym
import rospy
import roslaunch
import time
import numpy as np
import pdb
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan

from gym.utils import seeding
import pickle
import matplotlib.pyplot as plt

class GazeboTestEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboCircuit2TurtlebotLidar_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        high_a = np.array([0.3])
        low_o = np.array([0,0,0,0,0])
        high_o = np.array([np.inf,np.inf,np.inf,np.inf,np.inf])
        self.action_space = spaces.Box(low =-1*high_a,high=high_a) #F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.observation_space = spaces.Box(low=low_o, high=high_o)

        self.sim = True

        self._seed()

    def discretize_observation(self,data,new_ranges):
        discretized_ranges = []
        min_range = 0.2
        done = False
        mod = len(data.ranges)/new_ranges
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if data.ranges[i] == float ('Inf') or np.isinf(data.ranges[i]):
                    discretized_ranges.append(6)
                elif np.isnan(data.ranges[i]):
                    discretized_ranges.append(0)
                else:
                    discretized_ranges.append(data.ranges[i])
            if (min_range > data.ranges[i] > 0):
                done = True
        return discretized_ranges,done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        if self.sim:
            rospy.wait_for_service('/gazebo/unpause_physics')
            try:
                self.unpause()
            except (rospy.ServiceException) as e:
                print ("/gazebo/unpause_physics service call failed")

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = action[0]
        self.vel_pub.publish(vel_cmd)


        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=50)
            except:
                pass
	
	#print data.ranges
        if self.sim:
            rospy.wait_for_service('/gazebo/pause_physics')
            try:
                #resp_pause = pause.call()
                self.pause()
            except (rospy.ServiceException) as e:
                print ("/gazebo/pause_physics service call failed")
        
        state,done = self.discretize_observation(data,5)

        #pdb.set_trace() 

        if not done:
            if np.sum(action) < 0.1:
                reward = 5
            else:
                reward = 1
        else:
            reward = -200

        return state, reward, done, {}

    def _reset(self):

        # Resets the state of the environment and returns an initial observation.
        if self.sim:
            rospy.wait_for_service('/gazebo/reset_simulation')
            try:
                #reset_proxy.call()
                self.reset_proxy()
            except (rospy.ServiceException) as e:
                print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        if self.sim:
            rospy.wait_for_service('/gazebo/unpause_physics')
            try:
                #resp_pause = pause.call()
                self.unpause()
            except (rospy.ServiceException) as e:
                print ("/gazebo/unpause_physics service call failed")

        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=50)
            except:
                pass

        if self.sim:
            rospy.wait_for_service('/gazebo/pause_physics')
            try:
                #resp_pause = pause.call()
                self.pause()
            except (rospy.ServiceException) as e:
                print ("/gazebo/pause_physics service call failed")

        state,done = self.discretize_observation(data,5)

        return state

def plot(t, rewa, scor):
    font = {'family' : 'sens-serif',
    'weight' : 'bold',
    'size'   : 19}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)

    plt.figure(1)
    plt.subplot(211)
    plt.tight_layout()
    plt.plot(t, rewa)
    plt.title('Rewards')

    plt.subplot(212)
    plt.tight_layout()
    plt.plot(t, scor)
    plt.title('State Safety Scores')

    plt.tight_layout()
    plt.show()

def main():
        fname = "rob7:q_model5.txt"
        model = pickle.load(open(fname,"rb"))
        env = GazeboTestEnv()
        s = env.reset()

        T = 1000

        rewa = np.zeros((T,1)) # reward
        stat = np.zeros((T,1)) # state
        nsta = np.zeros((T,1)) # next state
        acti = np.zeros((T,1)) # action
        scor = np.zeros((T,1)) # reliability scores

        max_score = np.max(np.sum(model.vpl.KDD,axis = 0))

        for t in range(0,T):
            stat[t] = s
            scor[t] = np.sum(model.vpl.kernel.f(s,model.vpl.D))

            a = np.reshape(np.clip(model.get_pi(s),-env.high_a, env.high_a))

            s, r, done, _ = env.step(a)

            acti[a] = a
            rewa[t] = r
            nsta[t] = s

            if done:
                env.reset()
                
        t = np.linspace(0, T, T)
        plot(t, rewa, scor)




