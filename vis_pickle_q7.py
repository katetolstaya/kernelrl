
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
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
n = 100
num_files = 10
c1 = np.zeros((num_files,n))
c2 = np.zeros((num_files,n))
c3 = np.zeros((num_files,n))


#f =  open('../rlcore/exp_results/exp9_21multi/exp_9_21multi.pkl','rb')
f =  open('../rlcore/exp8_8multi.pkl','rb')
stats = cPickle.load(f)


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 19}
plt.rc('font', **font)
plt.rc('text', usetex=True)
#plt.figure(1)



nxindex = np.linspace(0, n/20, n)

colors = {0: 'lightcoral', 1: 'lightgreen', 2: 'lightblue', 3:'khaki', 4:'pink', 5:'silver', 6:'plum', 7:'tan', 8:'navajowhite', 9:'lavender'}


for file_num2 in range(1,num_files+1):
	

	#['all_metrics', 'all_episode_rewards', 'agent', 'interval_metrics', 'interval_rewards']
	test_loss = np.array(stats['exp'+str(file_num2)]['interval_metrics']['ACC Testing Loss'][0:n])
	normsq = np.array(stats['exp'+str(file_num2)]['interval_metrics']['ACC Regularized Testing Loss'][0:n])
	rewards2 = np.array(stats['exp'+str(file_num2)]['interval_metrics']['Testing Reward'])
	model_order = stats['exp'+str(file_num2)]['interval_metrics']['Model Order'][0:n]
	#print stats['exp'+str(file_num2)]['interval_rewards']['mean']
	#testing_reward = stats['exp'+str(file_num2)]['all_episode_rewards']['reward'][0:n]
	#print testing_reward

	
	#c[file_num2-1,:] = test_loss / np.sqrt(normsq)
	#c[file_num2-1,:] = model_order\

	#n_temp = len(stats['exp'+str(file_num2)]['all_episode_rewards']['reward'])

	#plt.plot(nxindex, c[file_num2-1,:], linewidth=1.0,color=(colors[file_num2-1]))


	rewards = np.array(stats['exp'+str(file_num2)]['all_episode_rewards']['reward'])

	c1[file_num2-1,:] =  signal.resample(rewards, num=n, window=10)
	c2[file_num2-1,:] =  model_order
	c3[file_num2-1,:] =  test_loss / np.sqrt(normsq)

d1 = np.average(c1, axis=0)
d2 = np.average(c2, axis=0)
d3 = np.average(c3, axis=0)
###############################

fig, ax = plt.subplots(1, 1)
fig.set_figheight(5.2)
fig.set_figwidth(6.47)

for file_num2 in range(1,num_files+1):
	plt.plot(nxindex , c1[file_num2-1,:], linewidth=1.0,color=(colors[file_num2-1]))
plt.plot(nxindex, d1, color='black', linewidth=2.0)
plt.xlabel('Training Steps ($10^5$)')
plt.axhline(y=90, color='limegreen', linestyle='-', linewidth=2.0)
plt.ylabel('Average Episode Reward')
ax.grid(color='k', linestyle='-', linewidth=0.25)
plt.tight_layout()
#plt.show()

fig.savefig('reward.png', dpi=200)

fig.savefig('reward.eps')

###############################
fig, ax = plt.subplots(1, 1)
fig.set_figheight(5.2)
fig.set_figwidth(6.47)

for file_num2 in range(1,num_files+1):
	plt.plot(nxindex , c2[file_num2-1,:], linewidth=1.0,color=(colors[file_num2-1]))
plt.plot(nxindex, d2, color='black', linewidth=2.0)
plt.xlabel('Training Steps ($10^5$)')
plt.ylabel('Model Order')
ax.grid(color='k', linestyle='-', linewidth=0.25)
plt.tight_layout()
#plt.show()

fig.savefig('modelorder.png', dpi=200)

fig.savefig('modelorder.eps')

################################
fig, ax = plt.subplots(1, 1)
fig.set_figheight(5.2)
fig.set_figwidth(6.47)

for file_num2 in range(1,num_files+1):
	plt.plot(nxindex , c3[file_num2-1,:], linewidth=1.0,color=(colors[file_num2-1]))
plt.plot(nxindex, d3, color='black', linewidth=2.0)
plt.xlabel('Training Steps ($10^5$)')
plt.ylabel('Normalized Bellman Error')
ax.grid(color='k', linestyle='-', linewidth=0.25)
plt.tight_layout()
#plt.show()

fig.savefig('normbellerr.png', dpi=200)

fig.savefig('normbellerr.eps')


