
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import pksarsa
#import kpolicy
#from function import KernelRepresentation


folder = 'exp_results/exp9_16multi/exp9_16multi_9'
n = 100
c = np.zeros((n,3))
for file_num in range(7,10):
	f =  open('../rlcore/'+folder+str(file_num)+'.pkl','rb')
	stats = cPickle.load(f)
	for file_num2 in range(1,n):
		c[file_num2-1,file_num-7] = np.average(stats['exp'+str(file_num2)]['interval_metrics']['Training Loss']) 


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)
#plt.figure(1)
fig, ax = plt.subplots(1, 1)
fig.set_figheight(12)
fig.set_figwidth(19.416)

d = np.average(c, axis=1)
nxindex = np.linspace(1, n*100000, n)

plt.plot(nxindex, c[:,0], linewidth=1.0,color='lightcoral')
plt.plot( nxindex, c[:,1], linewidth=1.0,color='lightgreen')
plt.plot(nxindex, c[:,2],color='lightblue',linewidth=1.0)
plt.plot(nxindex, d, color='black', linewidth=2.0)

plt.xlabel('Training Steps')
plt.ylabel('Test Bellman Error')
#plt.ylabel('Hilbert Norm of Q')
#plt.ylabel('Average V(s)')
ax.grid(color='k', linestyle='-', linewidth=0.5)


plt.show()

