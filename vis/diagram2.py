import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from math import e, sqrt, pi

font = {'family': 'serif',
        'weight': 'bold',
        'size': 15}
plt.rc('font', **font)
plt.rc('text', usetex=True)

# # Our 2-dimensional distribution will be over variables X and Y
# N = 200


# Sigma = 0.1

# def gauss(x, m, s):
#     return 1/(sqrt(2*pi)*s)*e**(-0.5*((x-m)/s)**2)

# # The distribution on the variables X, Y packed into pos.
# Z1 = gauss(X,mu1[0],Sigma) + gauss(X,mu1[1],Sigma) + gauss(X,mu1[2],Sigma)

# #Z2 = gauss(X,mu2[0],Sigma) 

# Z3 = gauss(X,mu3[0],Sigma) + gauss(X,mu3[1],Sigma) + gauss(X,mu3[2],Sigma) + gauss(X,mu3[3],Sigma) + gauss(X,mu3[4],Sigma)


# Z = gauss(X,mu1[0],Sigma) + gauss(X,mu1[1],Sigma) + gauss(X,mu1[2],Sigma) + gauss(X,mu3[4],Sigma) + gauss(X,mu3[0],Sigma) + gauss(X,mu3[1],Sigma) + gauss(X,mu3[2],Sigma)

# # Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
#ax = fig.gca(projection='3d')
fig, ax = plt.subplots(1, 1)


mu1 = np.random.uniform(low = 0.2, high = 0.5, size = (5,2))
mu2 = np.random.uniform(low = 0.5, high = 0.8, size = (3,2))
mu3 = np.random.uniform(low = 0, high = 0.6, size = (6,2))
mu4 = np.random.uniform(low = 0, high = 0.8, size = (4,2))


ax.plot(mu1[:,0] ,mu1[:,1], 'r8' , label='Policy 1', markersize =18.0)
ax.plot(mu2[:,0] ,mu2[:,1], 'bo' , label='Policy 2', markersize =18.0)
ax.plot(mu3[:,0] ,mu3[:,1], 'gs' , label='Policy 3', markersize =18.0)
ax.plot(mu4[:,0] ,mu4[:,1], 'm*' , label='Policy 4', markersize =18.0)

ax.set_yticklabels([])
ax.set_xticklabels([])

ax.set_xlim(0,1)
ax.set_ylim(0,1)

ax.legend( loc=1, borderaxespad=0.)
plt.xlabel('State 1')
plt.ylabel('State 2')
#ax.plot_surface(X, Y, Z, rstride=3, cstride=3,  antialiased=True,cmap=cm.viridis) #linewidth=1,


#cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# Adjust the limits, ticks and view angle

#ax.set_zticks(np.linspace(0,0.2,5))
#ax.view_init(27, -21)

plt.show()