
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
folder = 'exp9_91'
n = 92
f =  open('../rlcore/'+folder+'/kpolicy_model_'+str(n)+'.pkl','rb')
model = cPickle.load(f)

#print (ret['interval_metrics'].keys())

n_p = 20
n_v = 20
x = np.array(np.linspace(-1.2, 0.6, n_p)) #np.array(np.linspace(-10, 10, n_p))
y = np.array(np.linspace(-0.07, 0.07, n_v)) #np.array(np.linspace(-5, 5, n_v))
#z = 1
#v = 0

#v = np.zeros((n_p,n_v))
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

	
print np.average(b, axis=(0,1))	
	#a[i][j] = ret['agent'].model.Q.argmax(np.array([x[i],y[j]]))

#plt.imshow(a, interpolation='nearest')
#plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
fig, ax = plt.subplots(1, 1)

xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
# mc car
#plot = ax.pcolormesh(xv, yv, a, cmap='seismic', vmin = -1, vmax = 1)
plot = ax.pcolormesh(xv, yv, b, cmap='seismic', vmin = -200, vmax = 200)

# point mass
#plot = ax.pcolormesh(xv, yv, b, cmap='seismic', vmin = -1000, vmax = 1000)
#plot = ax.pcolormesh(xv, yv, b, cmap='seismic', vmin = -1000, vmax = 1000)
fig.colorbar(plot)
plt.xlabel('Position')
plt.ylabel('Velocity')
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
