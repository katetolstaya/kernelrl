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

# Our 2-dimensional distribution will be over variables X and Y
N = 200
X = np.linspace(0.25, 2.75, N)

mu1 = np.array([0.85, 1.0, 1.15])
#mu2 = np.array([])
mu3 = np.array([1.85, 2.0, 2.15,0.8, 1.5])

Sigma = 0.1

def gauss(x, m, s):
    return 1/(sqrt(2*pi)*s)*e**(-0.5*((x-m)/s)**2)

# The distribution on the variables X, Y packed into pos.
Z1 = gauss(X,mu1[0],Sigma) + gauss(X,mu1[1],Sigma) + gauss(X,mu1[2],Sigma)

#Z2 = gauss(X,mu2[0],Sigma) 

Z3 = gauss(X,mu3[0],Sigma) + gauss(X,mu3[1],Sigma) + gauss(X,mu3[2],Sigma) + gauss(X,mu3[3],Sigma) + gauss(X,mu3[4],Sigma)


Z = gauss(X,mu1[0],Sigma) + gauss(X,mu1[1],Sigma) + gauss(X,mu1[2],Sigma) + gauss(X,mu3[4],Sigma) + gauss(X,mu3[0],Sigma) + gauss(X,mu3[1],Sigma) + gauss(X,mu3[2],Sigma)

# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
#ax = fig.gca(projection='3d')
fig, ax = plt.subplots(1, 1)
ax.plot(X, Z ,'k',linewidth=4.0, label='Composed')
ax.plot(X, Z1 ,'r', label='$f_1(x)$')
#ax.plot(X, Z2 ,'g', label='$f_2(x)$')
ax.plot(X, Z3 ,'b', label='$f_3(x)$')


ax.plot(mu1,np.array([-1,-1,-1]), 'ro')
#ax.plot(mu2,np.array([-1,-1]), 'go')
ax.plot(mu3,np.array([-1,-1,-1,-1,-1]), 'bo')

ax.set_yticklabels([])
ax.set_xticklabels([])

ax.legend( loc=1, borderaxespad=0.)
plt.xlabel('State')
plt.ylabel('Action')
#ax.plot_surface(X, Y, Z, rstride=3, cstride=3,  antialiased=True,cmap=cm.viridis) #linewidth=1,


#cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# Adjust the limits, ticks and view angle
#ax.set_zlim(-0.15,0.2)
#ax.set_zticks(np.linspace(0,0.2,5))
#ax.view_init(27, -21)

plt.show()