
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
n = 100

c = np.zeros((n,1))

for file_num in range (1,n):
	f =  open('../rlcore/'+folder+'/kpolicy_model_'+str(file_num)+'.pkl','rb')
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

	
	c[file_num][0] = np.average(b, axis=(0,1))	


plt.figure(1)

print (np.shape(c))
print(np.shape(np.linspace(1, n*10000, n)))

plt.plot(np.linspace(1, n, n), c)
plt.title('Interval Average Reward')

plt.show()

