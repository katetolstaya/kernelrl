
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import pksarsa
#import kpolicy
#from function import KernelRepresentation


#folder = 'exp_results/exp9_16multi/exp9_16multi_9'
#n = 100
#c = np.zeros((n,3))
#for file_num in range(7,10):
#	f =  open('../rlcore/'+folder+str(file_num)+'.pkl','rb')
#	stats = cPickle.load(f)
#	for file_num2 in range(1,n):
#		c[file_num2-1,file_num-7] = np.average(stats['exp'+str(file_num2)]['interval_metrics']['Training Loss']) 

#folder = 'exp_results/exp9_16multi/exp9_16multi_9'
n = 50
c = np.zeros((10,n))

#f =  open('../rlcore/exp_results/exp9_21multi/exp_9_21multi.pkl','rb')
f =  open('../rlcore/exp9_212test.pkl','rb')
stats = cPickle.load(f)

dict_str = 'exp10_33'

print stats[dict_str]['interval_metrics'].keys()
print stats[dict_str]['interval_rewards'].keys()
print stats[dict_str]['all_episode_rewards'].keys()
print stats[dict_str]['all_metrics'].keys()
print stats[dict_str].keys()


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 19}

plt.rc('font', **font)
plt.rc('text', usetex=True)
#plt.figure(1)
fig, ax = plt.subplots(1, 1)
fig.set_figheight(12)
fig.set_figwidth(19.416)


colors = {0: 'lightcoral', 1: 'lightgreen', 2: 'lightblue', 3:'khaki', 4:'pink', 5:'silver', 6:'plum', 7:'tan', 8:'navajowhite', 9:'lavender'}

nxindex = np.linspace(1, n/10, n)

for i in range(1,11):
	for j in range(1,51):
		dict_str = 'exp' + str(i) + '_' + str(j) 
		#print 'PolicyFile = exp9_9' + str(i) + '/kpolicy_model_' +str(j)
		testing_reward = np.average(stats[dict_str]['all_episode_rewards']['reward'])
		#print stats[dict_str]['interval_rewards']['mean']
		c[i-1,j-1] = testing_reward
	plt.plot(nxindex, c[i-1,:], linewidth=1.0,color=(colors[i-1]))
	

d = np.average(c, axis=0)

plt.plot(nxindex, d, color='black', linewidth=2.0)



plt.xlabel('Training Steps ($10^5$)')
#plt.ylabel('Normalized Bellman Error')
plt.axhline(y=90, color='limegreen', linestyle='-', linewidth=2.0)
plt.ylabel('Average Episode Reward')
#plt.ylabel('Bellman Error')
#plt.ylabel('Model Order')
#plt.ylabel('Hilbert Norm of Q')
#plt.ylabel('Average V(s)')
ax.grid(color='k', linestyle='-', linewidth=0.5)


plt.show()

