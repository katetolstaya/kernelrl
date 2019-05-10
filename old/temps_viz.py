import pickle
import numpy as np
import matplotlib.pyplot as plt

model = pickle.load(open('temps2503.pkl','rb'))


n = 2000
dn = 5

data = np.zeros((n/dn,n/dn))
data2 = np.zeros((n/dn,n/dn))

for i in range(0,n,dn):
	for j in range(0,n,dn):
		data[i/dn,j/dn] = model.val(np.array([i,j]))
		data2[i/dn,j/dn] = model.var(np.array([i,j]))

f, axarr = plt.subplots(2)
pos1 = axarr[0].imshow(data, cmap='coolwarm', interpolation='nearest')
axarr[0].set_title('Value')
f.colorbar(pos1, ax=axarr[0])
pos2 = axarr[1].imshow(data2, cmap='Reds', interpolation='nearest', vmax=100)
axarr[1].set_title('Variance')
f.colorbar(pos2, ax=axarr[1])
plt.show()