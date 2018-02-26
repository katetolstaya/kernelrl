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

    def __init__(self, fname):
        # Launch the simulation with the given launchfile name
        self.fname = fname
        #self.fname = "GazeboCircuit2TurtlebotLidar_v0.launch"
        #self.fname = "GazeboCircuit2TurtlebotLidar_v0.launch"
        #self.fname = "GazeboMazeTurtlebotLidar_v0.launch"
        gazebo_env.GazeboEnv.__init__(self, self.fname)
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.high_a = np.array([0.3])
        low_o = np.array([0,0,0,0,0])
        high_o = np.array([np.inf,np.inf,np.inf,np.inf,np.inf])
        self.action_space = spaces.Box(low =-1*self.high_a,high=self.high_a) #F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.observation_space = spaces.Box(low=low_o, high=high_o)

        self.sim = True

        self._seed()
        print "Initialized"

    def discretize_observation(self,data,new_ranges):
        discretized_ranges = []
        min_range = 0.20
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
        print "Reset"

        return state

def main():

    launchf = "GazeboCircuit2TurtlebotLidar_v0.launch"
    #launchf = "GazeboCircuit2TurtlebotLidar_v0.launch"
    #launchf = "GazeboMazeTurtlebotLidar_v0.launch"
    #launchf = "GazeboRoundTurtlebotLidar_v0.launch"
    #launchf = "GazeboPillarsTurtlebotLidar_v0.launch"

    flist = open('combos/results2_circuit2.txt','wb')

    model_names = [] #["exp20/rob20_model23.txt", "exp21/rob21_model63.txt", "exp22/rob22_model61.txt"]
    N = 14
    T = 1000

    if len(model_names) is 0:
        for i in range(0,N+1):
            model_names.append("combos/policy"+str(i)+".txt")


    env = GazeboTestEnv(launchf)

    for fname in model_names:

        model = pickle.load(open(fname,"rb"))
        print (fname)

        s = env.reset()
        rewa = np.zeros((T,1)) # reward
        loss = np.zeros((T,1)) # reward
        n_crashes = 0

        for t in range(0,T):
            noise = 0 #np.random.normal(0,0.2,1)
            a = np.reshape(np.clip(model.get_pi(s),-env.high_a, env.high_a), (-1,))
            qs = model.get_q(s,a)
            s, r, done, _ = env.step(a)
            rewa[t] = r
            loss[t] = (qs - r - 0.99 * model.get_v(s)) ** 2

            if done:
                env.reset()
                n_crashes = n_crashes + 1
                #print(t)

        flist.write(str(fname) + ", " + str(np.sum(rewa)) + ", " +  str(np.sum(loss)) + '\n')

if __name__ == "__main__":
    main()
















