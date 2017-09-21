
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

f =  open('../rlcore/exp_results/exp9_21multi/exp_9_21multi.pkl','rb')
stats = cPickle.load(f)


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)
#plt.figure(1)
fig, ax = plt.subplots(1, 1)
fig.set_figheight(12)
fig.set_figwidth(19.416)


nxindex = np.linspace(1, n*10000, n)

colors = {0: 'lightcoral', 1: 'lightgreen', 2: 'lightblue', 3:'khaki', 4:'pink', 5:'silver', 6:'plum', 7:'tan', 8:'navajowhite', 9:'lavender'}


for file_num2 in range(1,11):
	#['all_metrics', 'all_episode_rewards', 'agent', 'interval_metrics', 'interval_rewards']
	test_loss = np.array(stats['exp'+str(file_num2)]['interval_metrics']['Testing Loss'][0:n])
	normsq = np.array(stats['exp'+str(file_num2)]['interval_metrics']['Regularized Testing Loss'][0:n])
	model_order = stats['exp'+str(file_num2)]['interval_metrics']['Model Order'][0:n]

	c[file_num2-1,:] =   test_loss / np.sqrt(normsq)

	plt.plot(nxindex, c[file_num2-1,:], linewidth=1.0,color=(colors[file_num2-1]))

d = np.average(c, axis=0)
plt.plot(nxindex, d, color='black', linewidth=2.0)




plt.xlabel('Training Steps')
plt.ylabel('Test Bellman Error')
#plt.ylabel('Hilbert Norm of Q')
#plt.ylabel('Average V(s)')
ax.grid(color='k', linestyle='-', linewidth=0.5)


plt.show()

