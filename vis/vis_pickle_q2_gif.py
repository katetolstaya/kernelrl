
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import pksarsa
#import kpolicy
#from function import KernelRepresentation


#method = 'SCGD'

import imageio
images = []
folder = 'exp6_20'


for n in xrange(1,50,3):
	method = 'DEFAULT'
	f =  open('../rlcore/'+folder+'/kpolicy_model_'+str(n)+'.pkl','rb')
	ret = cPickle.load(f)


	x = np.array(np.linspace(-50, 50, 20))
	y = np.array(np.linspace(-5, 5, 20))
	z = 1
	v = 0

	v = np.zeros((np.shape(x)[0],np.shape(y)[0]))
	a = np.zeros((np.shape(x)[0],np.shape(y)[0]))
	for i in range(0,np.shape(x)[0]):
	    for j in range (0,np.shape(x)[0]):
	
		#v[i][j] = (ret(np.array([x[i],y[j],0])));
		a[i][j] = ret.argmax(np.array([x[i],y[j]]))

	#plt.imshow(a, interpolation='nearest')
	#plt.show()

	#fig = plt.figure()
	#ax = fig.add_subplot(111, projection='3d')
	fig, ax = plt.subplots(1, 1)

	xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

	plot = ax.pcolormesh(xv, yv, a, cmap='seismic',vmin=-2., vmax=2.,)
	fig.colorbar(plot)
	plt.xlabel('Position')
	plt.ylabel('Velocity')
	plt.title('Iteration ' + str(n))
	#plt.show()
	plt.savefig('myfig')
	img = imageio.imread('myfig.png')
	for m in xrange(1,10):
    		images.append(img)

for m in xrange(1,10):
	images.append(img)
imageio.mimsave('../rlcore/'+folder+'/policies.gif', images)

#ax.pcolormesh(X, Y, h)

#--------------------------------------------

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
#ax.plot_wireframe(xv, yv, v, color='b')
#plt.show()

#---------------------------------------------

#test_loss = ret[method]['interval_metrics']['Testing Loss']
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
