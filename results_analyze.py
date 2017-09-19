import re
import numpy as np
import matplotlib.pyplot as plt

filename = 'exp9_1910.txt'
#filename = 'exp_results/exp9_15multi/exp9_9.txt'
#exp9_195.txt'
#filename = 'exp7_25.txt'
interval = 1000
pattern = re.compile("episode_rewards: ([0-9]*.[0-9]*)")
pattern2 = re.compile("^([0-9]+)$")
pattern3 = re.compile("Model Order: ([0-9]*.[0-9]*)")
pattern4 = re.compile("Training Loss: ([0-9]*.[0-9]*)")
#episode_rewards: 36.938
n = 100000
a = np.zeros((n,))
b = np.zeros((5*n,))
c = np.zeros((n,))
d = np.zeros((n,))
j = 0
m = 0
k = 0
p = 0
for i, line in enumerate(open(filename)):
    for match in re.finditer(pattern, line):
	a[k] = float(match.groups()[0])
	k = k+1

for i, line in enumerate(open(filename)):
    for match in re.finditer(pattern2, line):
	b[j] = float(match.groups()[0])
        j = j+1

for i, line in enumerate(open(filename)):
    for match in re.finditer(pattern3, line):
	c[m] = float(match.groups()[0])
	m = m+1

for i, line in enumerate(open(filename)):
    for match in re.finditer(pattern4, line):
	d[p] = float(match.groups()[0])
	p = p+1


window =1
#k = 100


window2 = window*1
a = np.convolve(a[0:k-1], np.ones((window,))/window, mode='valid')
b = np.convolve(b[0:j-1], np.ones((window2,))/window2, mode='valid')
c = c[0:m-1]
d = d[0:p-1]


plt.figure(1)
plt.subplot(411)

plt.plot(np.linspace(0, 1000*len(a), len(a)), a)
plt.title('Interval Average Reward')
#plt.xlabel('Training Steps')
plt.ylabel('Average Reward, (w='+str(window)+')')
#plt.xlim([0,10000000])

plt.subplot(412)

plt.plot(np.linspace(0, len(b), len(b)), b)
plt.title('Episode Length')
#plt.xlabel('Training Episodes')
plt.ylabel('Episode Length (steps),(w='+str(window2)+')')


plt.subplot(413)

plt.plot(np.linspace(0, len(c)*200, len(c)), c)
plt.title('Model Order')
#plt.xlabel('Training Steps')
plt.ylabel('Model Order')

plt.subplot(414)
plt.plot(np.linspace(0, len(d)*200, len(d)), d)
plt.title('Training Error')
#plt.xlabel('Training Steps')
plt.ylabel('Training Error')
plt.show()
