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

def plot(t, rewa, scor):
    #font = {'family' : 'sens-serif',
    #'weight' : 'bold',
    #'size'   : 19}
    #plt.rc('font', **font)
    #plt.rc('text', usetex=True)

    plt.figure(1)
    plt.subplot(211)
    plt.tight_layout()
    plt.plot(t, rewa)
    plt.title('Rewards')

    plt.subplot(212)
    plt.tight_layout()
    plt.plot(t, scor)
    plt.title('State Safety Scores')

    plt.show()

def main():
    #fname = "comod2.txt" # combined
    #fname = "comod4.txt" # combined
    #fname = "rob22_model50.txt"
    #fname = "robot_results/rob15_model25.txt" # circuit 2 
    #fname = "robot_results/rob16_model49.txt" # circuit 2 
    #fname = "robot_results/rob14_model54.txt" # circuit 1
    #fname = "exp18/rob16_model74.txt" # maze
    #fname = "exp20/rob20_model23.txt" # round
    #fname = "rob19_2_model20.txt" # pillars
    # 21 maze
    # 22 circuit 1
    # 23 circuit 2

    #fname = "exp22/rob22_model55.txt"
    #fname = "exp22/rob22_2_model15.txt"
    #"exp23/rob23_model20.txt"

    #fname = "exp20/rob20_model23.txt" # round - GOOD
    #fname = "exp22/rob22_2_model15.txt" #rob22_model61.txt" # circuit 1 - BAD
    #fname = "exp22/rob22_model50.txt" #rob22_model61.txt" # circuit 1 - BAD
    #fname = "exp23/rob23_3_model3.txt" # circuit 2 - GOOD
    #fname = "exp21/rob21_model63.txt" # maze - GOOD
    fname = "combos/policy14.txt"
    fname = "lst.txt"



    #launchf = "GazeboRoundTurtlebotLidar_v0.launch"
    launchf = "GazeboCircuitTurtlebotLidar_v0.launch"
    #launchf = "GazeboMazeTurtlebotLidar_v0.launch"
    #launchf = "GazeboRoundTurtlebotLidar_v0.launch"
    #launchf = "GazeboPillarsTurtlebotLidar_v0.launch"

    #"rob15_model25.txt"
    #"rob9_model36.txt" #"combined_model2.txt" #
    model = pickle.load(open(fname,"rb"))
    env = GazeboTestEnv(launchf)
    s = env.reset()

    T = 10000

    rewa = np.zeros((T,1)) # reward
    #stat = np.zeros((T,1)) # state
    #nsta = np.zeros((T,1)) # next state
    acti = np.zeros((T,1)) # action
    scor = np.zeros((T,1)) # reliability scores

    max_score = np.max(np.sum(model.vpl.KDD,axis = 0))
    n_crashes = 0

    for t in range(0,T):
        #stat[t] = s
        scor[t] = np.sum(model.vpl.kernel.f(model.vpl.D,np.reshape(s,(1,5))))

        noise = 0 #np.random.normal(0,0.2,1)

        a = np.reshape(np.clip(model.get_pi(s)+noise,-env.high_a, env.high_a), (-1,))

        s, r, done, _ = env.step(a)

        acti[t] = a
        rewa[t] = r
        #nsta[t] = s

        if done:
            env.reset()
            n_crashes = n_crashes + 1
            print(t)

    env._close()
    print ("Sum of rewards: " + str(np.sum(rewa)) + " for " + env.fname + " using policy " + fname)
    print ("number of crashes: " + str(n_crashes))

    t = np.linspace(0, T, T)
    plot(t, rewa, scor)



if __name__ == "__main__":
    main()

# 10k steps::::

# Sum of rewards: 43199.0 for GazeboCircuit2TurtlebotLidar_v0.launch using policy robot_results/rob15_model25.txt
# Sum of rewards: 34037.0 for GazeboCircuitTurtlebotLidar_v0.launch using policy robot_results/rob15_model25.txt

# Sum of rewards: 38035.0 for GazeboCircuitTurtlebotLidar_v0.launch using policy robot_results/rob14_model54.txt
# Sum of rewards: 29062.0 for GazeboCircuit2TurtlebotLidar_v0.launch using policy robot_results/rob14_model54.txt

# Sum of rewards: 40607.0 for GazeboCircuit2TurtlebotLidar_v0.launch using policy combined_model9.txt
# Sum of rewards: 34596.0 for GazeboCircuitTurtlebotLidar_v0.launch using policy combined_model9.txt

# Sum of rewards: 36314.0 for GazeboCircuitTurtlebotLidar_v0.launch using policy combined_model10.txt


#5k
# Sum of rewards: 14744.0 for GazeboMazeTurtlebotLidar_v0.launch using policy robot_results/rob14_model54.txt - 21 crashes
# Sum of rewards: 14823.0 for GazeboMazeTurtlebotLidar_v0.launch using policy combined_model10.txt - 13 crashes
# Sum of rewards: 1089.0 for GazeboMazeTurtlebotLidar_v0.launch using policy robot_results/rob16_model49.txt - 51 crashes!!

# Sum of rewards: 15677.0 for GazeboMazeTurtlebotLidar_v0.launch using policy combined_model10.txt - number of crashes: 7

# Sum of rewards: 12543.0 for GazeboCircuitTurtlebotLidar_v0.launch using policy combined_model10.txt  - number of crashes: 13

# Sum of rewards: 14675.0 for GazeboCircuit2TurtlebotLidar_v0.launch using policy combined_model10.txt - number of crashes: 9

# Sum of rewards: 16278.0 for GazeboCircuit2TurtlebotLidar_v0.launch using policy robot_results/rob16_model49.txt - number of crashes: 2



















