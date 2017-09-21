
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import pksarsa
#import kpolicy
#from function import KernelRepresentation


#method = 'SCGD'
method = 'DEFAULT'
#f =  open('../rlcore/exp6_20_planar.pkl','rb')

#f =  open('exp_6_29_qplanar.pkl','rb')
#ret = cPickle.load(f)[method]

#folder = 'exp9_75'
#n = 5000
#folder = 'exp_results/exp9_16multi/exp9_97'
folder = 'exp9_91'
n = 3
f =  open('../rlcore/'+folder+'/kpolicy_model_'+str(n)+'.pkl','rb')
model = cPickle.load(f)


n_p = 20
n_v = 20
x = np.array(np.linspace(-1.2, 0.6, n_p)) #np.array(np.linspace(-10, 10, n_p))
y = np.array(np.linspace(-0.07, 0.07, n_v)) #np.array(np.linspace(-5, 5, n_v))

a = np.zeros((n_p,n_v))
b = np.zeros((n_p,n_v))
for i in range(0,n_p):
    for j in range (0,n_v):
        #v[i][j] = (ret(np.array([x[i],y[j],0])));
	sta = np.array([x[i],y[j]])
	act = model.argmax(sta)
	a[i][j] = act
	sa = np.concatenate((np.reshape(sta,(1,-1)), np.reshape(act,(1,-1))),axis=1)

	#print (x[0])
	b[i][j] = model(sa)


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

plt.rc('font', **font)

fig, ax = plt.subplots(1, 1)
fig.set_figheight(13)
fig.set_figwidth(21.034)

print (np.max(b))
print (np.min(b))

xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
# mc car
plot = ax.pcolormesh(xv, yv, a, cmap='RdBu_r', vmin = -1, vmax = 1)
#plot = ax.pcolormesh(xv, yv, b, cmap='RdBu_r', vmin = -110, vmax = 110)

ax.plot(model.D[:,0], model.D[:,1],'wo')
ax.set_xlim((-1.2, 0.6))
ax.set_ylim((-0.07, 0.07))
#ax.grid(color='k', linestyle='-', linewidth=1)
fig.colorbar(plot)
plt.xlabel('Position (m)')
plt.ylabel('Velocity (m/s)')
plt.show()


