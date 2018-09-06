
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

folder = 'exp8_10'
n = 4
f =  open('../rlcore/'+folder+'/kpolicy_pi_'+str(n)+'.pkl','rb')
model = cPickle.load(f)

#print (ret['interval_metrics'].keys())

n_p = 20
n_v = 20
x = np.array(np.linspace(-10, 10, n_p))
y = np.array(np.linspace(-10, 10, n_v))
#z = 1
#v = 0

#v = np.zeros((n_p,n_v))
a = np.zeros((n_p,n_v,2))
b = np.zeros((n_p,n_v,2))
for i in range(0,n_p):
    for j in range (0,n_v):
        #v[i][j] = (ret(np.array([x[i],y[j],0])));
	#a[i][j] = model.argmax(np.array([x[i],y[j]]))
	sta = np.array([x[i],y[j]])
	#act = model.argmax(sta)
	#sa = np.concatenate((np.reshape(sta,(1,-1)), np.reshape(act,(1,-1))),axis=1)

	b[i][j] = model(sta)[0]
	

	
	
	#a[i][j] = ret['agent'].model.Q.argmax(np.array([x[i],y[j]]))

#plt.imshow(a, interpolation='nearest')
#plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
fig, ax = plt.subplots(1, 1)

xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

plot = ax.pcolormesh(xv, yv, b[:,:,0], cmap='seismic', vmin = -2, vmax = 2)
fig.colorbar(plot)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.show()

fig, ax = plt.subplots(1, 1)
plot = ax.pcolormesh(xv, yv, b[:,:,1], cmap='seismic', vmin = -2, vmax = 2)
fig.colorbar(plot)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.show()


#ax.pcolormesh(X, Y, h)

#--------------------------------------------

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
#ax.plot_wireframe(xv, yv, v, color='b')
#plt.show()

#---------------------------------------------

#test_loss = ret['interval_metrics']['Training Loss']
#window = 1000
#test_loss = np.convolve(test_loss, np.ones((window,))/window, mode='valid')

#test_loss = ret[method]['interval_metrics']['Model Order']
#amax = np.argmax(test_loss)
#test_loss[amax] = 0
#amax = np.argmax(test_loss)
#test_loss[amax] = 0
#plt.plot(np.linspace(0, 50000, np.shape(test_loss)[0]), test_loss)
#plt.ylabel('some numbers')
#plt.title(method + ' Test Loss')
#plt.xlabel('Steps')
#plt.ylabel('Testing Loss')
#plt.show()
