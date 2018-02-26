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

class RobotTestEnv():

    def __init__(self, fname, outname):

        self.model = pickle.load(open(fname,"rb"))
        self.outfile = open(outname,'wb')
        rospy.init_node('rlcore', anonymous=True)
        
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        rospy.Subscriber('/scan', LaserScan, self.callback)

        self.high_a = np.array([0.3])

        self.t = 0
        self.dt = 0.1
        #low_o = np.array([0,0,0,0,0])
        #high_o = np.array([np.inf,np.inf,np.inf,np.inf,np.inf])
        #self.action_space = spaces.Box(low =-1*self.high_a,high=self.high_a) #F,L,R
        #self.reward_range = (-np.inf, np.inf)
        #self.observation_space = spaces.Box(low=low_o, high=high_o)

        print "Initialized"
        rospy.spin()

    def discretize_observation(self,data,new_ranges):
        discretized_ranges = []
        min_range = 0.18
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

    def callback(self, data):

        s,done = self.discretize_observation(data,5)
        vel_cmd = Twist() # assuming this is initialized with 0

        if not done:
            reward = 1
            action = np.reshape(np.clip(self.model.get_pi(s),-self.high_a, self.high_a), (-1,))
            vel_cmd.linear.x = 0.15
            vel_cmd.angular.z = action[0]
        else:
            reward = -200

        score = np.sum(self.model.vpl.kernel.f(self.model.vpl.D,np.reshape(s,(1,5))))
        self.outfile.write(str(self.t) + ", " + str(reward) + ", " +  str(score) + '\n')
        self.t = self.t + self.dt

        self.vel_pub.publish(vel_cmd)

def main():
    fname = "combos/policy14.txt"
    outname = "combos/realrobot.txt"
    env = RobotTestEnv(fname, outname)


if __name__ == "__main__":
    main()


















