import cPickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ret = np.load('table_tab2_427.npy')

max_a = 2
max_p = 10
max_v = 5

var_a = 0.1
var_p = 1
var_v = 0.5

n_a = 5 #2*int(max_a/var_a)+1
n_v = 20 #2*int(max_v/var_v)+1
n_p = 20 #2*int(max_p/var_p)+1


x = np.array(np.linspace(-10, 10, n_p))
y = np.array(np.linspace(-5, 5, n_v))
#z = 1
#v = 0

#v = np.zeros((np.shape(x)[0],np.shape(y)[0]))
a = np.zeros((n_p, n_v))
b = np.zeros((n_p, n_v))
for i in range(0,n_p-1):
    for j in range (0,n_v-1):
	#v[i][j] = (ret(np.array([x[i],y[j],0])));
	a[i][j] = np.argmax(ret[i,j,:]) - max_a
	b[i][j] = ret[i,j,:].max()
	print b[i][j]

	#b = ret[i,j,:]
        #a[i][j] = np.array([np.random.choice(np.flatnonzero(b == b.max()))-max_a]) 

fig, ax = plt.subplots(1, 1)
xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
plot = ax.pcolormesh(xv, yv, a, cmap='seismic',vmin=-2., vmax=2.,)
fig.colorbar(plot)
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.show()

fig2, ax2 = plt.subplots(1, 1)
plot2 = ax2.pcolormesh(xv, yv, b, cmap='seismic')#, vmin = 0, vmax = 1000)
fig2.colorbar(plot2)
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.show()

