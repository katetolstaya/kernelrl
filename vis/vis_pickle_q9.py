import cPickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
import sys
import re
import pdb
#################################
font = {'family': 'normal',
        'weight': 'bold',
        'size': 19}
plt.rc('font', **font)
plt.rc('text', usetex=False)

interval = 1000
pattern1 = re.compile(r"episode_rewards: ([\-]?\d+\.\d+)")
pattern2 = re.compile(r"^([0-9]+)$")
pattern3 = re.compile(r"Model Order: ([0-9]*.[0-9]*)")
pattern4 = re.compile(r"Training Loss: (-?[0-9]*.[0-9]*)")
pattern5 = re.compile(r"Testing Loss: (-?[0-9]*.[0-9]*)")
pattern6 = re.compile(r"Testing Reward: (-?[0-9]*.[0-9]*)")
pattern7 = re.compile(r'(\d+) steps performed')  ## steps
#################################

save_flag = True

# p3
fdir = 'tac/p3/'
save_exp= 'p_'
all_names = 'rlcorejob.o44300  rlcorejob.o44303  rlcorejob.o44353  rlcorejob.o44356 rlcorejob.o44301  rlcorejob.o44304  rlcorejob.o44354 rlcorejob.o44302  rlcorejob.o44352  rlcorejob.o44355'
num_files = 10

# episode_rewards: 36.938
n = 500
a = np.zeros((n, num_files))
b = np.zeros((n, num_files))
c = np.zeros((n, num_files))
d = np.zeros((n, num_files))
e = np.zeros((n, num_files))

fnum = 0
for fname in all_names.split():
    filename = fdir + fname

    j = 0
    m = 0
    k = 0
    p = 0
    q = 0

    for i, line in enumerate(open(filename)):
        if k >= n:
            break

        for match in re.finditer(pattern1, line):  # test reward - change back to 6
            a[k, fnum] = float(match.groups()[0])
            k = k + 1
        for match in re.finditer(pattern3, line):  # Model order
            b[m, fnum] = float(match.groups()[0])
            m = m + 1
        for match in re.finditer(pattern5, line):  # test error
            c[j, fnum] = float(match.groups()[0])
            j = j + 1
        for match in re.finditer(pattern4, line):  # train error
            d[p, fnum] = float(match.groups()[0])
            p = p + 1
        for match in re.finditer(pattern7, line):  # step index
            e[q, fnum] = int(match.groups()[0])
            q = q + 1

    fnum = fnum + 1

colors = {0: 'lightcoral', 1: 'lightgreen', 2: 'lightblue', 3: 'khaki', 4: 'pink', 5: 'silver', 6: 'plum', 7: 'tan',
          8: 'navajowhite', 9: 'lavender'}

e = e/np.power(10,5)

a_av = np.average(a, axis=1)
b_av = np.average(b, axis=1)
c_av = np.average(c, axis=1)
d_av = np.average(d, axis=1)
###############################

#pdb.set_trace()


fig, ax = plt.subplots(1, 1)
fig.set_figheight(5.2)
fig.set_figwidth(6.47)

for file_num2 in range(1, num_files + 1):
    plt.plot(e[:, file_num2 - 1], a[:, file_num2 - 1], linewidth=1.0, color=(colors[file_num2 - 1]))

plt.plot(e[:, 0], a_av, color='black', linewidth=2.0)
plt.xlabel('Training Steps ($10^5$)')
#plt.axhline(y=0, color='limegreen', linestyle='-', linewidth=2.0)
plt.ylabel('Average Episode Reward')
ax.grid(color='k', linestyle='-', linewidth=0.25)
plt.tight_layout()

if save_flag:
    fig.savefig(fdir + save_exp + 'reward.png', dpi=200)
    fig.savefig(fdir + save_exp + 'reward.eps')
else:
    plt.show()

###############################
fig, ax = plt.subplots(1, 1)
fig.set_figheight(5.2)
fig.set_figwidth(6.47)

for file_num2 in range(1, num_files + 1):
    plt.plot(e[:, file_num2 - 1], b[:, file_num2 - 1], linewidth=1.0, color=(colors[file_num2 - 1]))

plt.plot(e[:, 0], b_av, color='black', linewidth=2.0)
plt.xlabel('Training Steps ($10^5$)')
plt.ylabel('Model Order')
ax.grid(color='k', linestyle='-', linewidth=0.25)
plt.tight_layout()

if save_flag:
    fig.savefig( fdir +  save_exp + 'modelorder.png', dpi=200)
    fig.savefig( fdir + save_exp + 'modelorder.eps')
else:
    plt.show()

################################
fig, ax = plt.subplots(1, 1)
fig.set_figheight(5.2)
fig.set_figwidth(6.47)

for file_num2 in range(1, num_files + 1):
    plt.plot(e[:, file_num2 - 1], d[:, file_num2 - 1,], linewidth=1.0, color=(colors[file_num2 - 1]))

plt.plot(e[:, 0], d_av, color='black', linewidth=2.0)
plt.xlabel('Training Steps ($10^5$)')
plt.ylabel('Bellman Error')
ax.grid(color='k', linestyle='-', linewidth=0.25)
plt.tight_layout()

if save_flag:
    fig.savefig( fdir +  save_exp + 'normbellerr.png', dpi=200)
    fig.savefig( fdir + save_exp + 'normbellerr.eps')
else:
    plt.show()
