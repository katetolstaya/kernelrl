
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#method = 'SCGD'
method = 'DEFAULT'
f =  open('exp_5_31_kq.pkl','rb')
ret = cPickle.load(f)


#x = np.array(np.linspace(-10, 10, 20))
#y = np.array(np.linspace(-10, 10, 20))
#z = 0
#
#v = np.zeros((np.shape(x)[0],np.shape(x)[0]))
#
#for i in range(0,np.shape(x)[0]):
#    for j in range (0,np.shape(x)[0]):
#        v[i][j] = -(ret[method]['agent'].model.V(np.array([x[i],y[j],z,1,0,0,0])));
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
#ax.plot_wireframe(xv, yv, v, color='b')
#plt.show()

#test_loss = ret[method]['interval_metrics']['Testing Loss']
test_loss = ret[method]['interval_metrics']['Model Order']
#amax = np.argmax(test_loss)
#test_loss[amax] = 0
#amax = np.argmax(test_loss)
#test_loss[amax] = 0
plt.plot(np.linspace(0, 50000, np.shape(test_loss)[0]), test_loss)
plt.ylabel('some numbers')
plt.title(method + ' Test Loss')
plt.xlabel('Steps')
plt.ylabel('Testing Loss')
plt.show()
