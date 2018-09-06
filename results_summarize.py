import re
import numpy as np
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]
#filename = 'exp11_6_5.txt'
#filename = 'exp11_2_4.txt'
if len(sys.argv) >=3 :
    window = int(sys.argv[2])
else:
    window = 1
# filename = 'exp8_8multi.txt'
# filename = 'exp_results/exp9_21multi/exp9_21multi.txt'
# filename = 'exp_results/exp9_16multi/exp9_9.txt'
# exp9_195.txt'
# filename = 'exp7_25.txt'
interval = 1000
pattern1 = re.compile(r"episode_rewards: ([\-]?\d+\.\d+)")
pattern2 = re.compile(r"^([0-9]+)$")
pattern3 = re.compile(r"Model Order: ([0-9]*.[0-9]*)")
pattern4 = re.compile(r"Training Loss: (-?[0-9]*.[0-9]*)")
pattern5 = re.compile(r"Testing Loss: (-?[0-9]*.[0-9]*)")
pattern6 = re.compile(r"Testing Reward: (-?[0-9]*.[0-9]*)")
# episode_rewards: 36.938
n = 100000
a = np.zeros((n,))
b = np.zeros((5 * n,))
c = np.zeros((n,))
d = np.zeros((n,))
f = np.zeros((n,))
j = 0
m = 0
k = 0
p = 0
q = 0
for i, line in enumerate(open(filename)):
    for match in re.finditer(pattern6, line): # test reward
        a[k] = float(match.groups()[0])
        k = k + 1
    for match in re.finditer(pattern3, line): # Model order
        b[m] = float(match.groups()[0])
        m = m + 1
    for match in re.finditer(pattern5, line):  # test error
        c[j] = float(match.groups()[0])
        j = j + 1
    for match in re.finditer(pattern4, line): # train error
        d[p] = float(match.groups()[0]) 
        p = p + 1
    for match in re.finditer(pattern1, line): # train error
        f[q] = float(match.groups()[0]) 
        q = q + 1

#a = np.convolve(a[0:k - 1], np.ones((window,)) / window, mode='valid')
#b = np.convolve(b[0:m - 1], np.ones((window,)) / window, mode='valid')
#c = np.convolve(c[0:j - 1], np.ones((window,)) / window, mode='valid')
#d = np.convolve(d[0:p - 1], np.ones((window,)) / window, mode='valid')

a = a[0:k - 1]
b = b[0:m - 1]
c = c[0:j - 1]
d = d[0:p - 1]
f = f[0:q - 1]

N = 20

print "Model Order/Training Loss/Training Rewards/Testing Rewards/Testing Loss"
print str(round(np.mean(b[-N:]),2)) + " / " + str(round(np.mean(d[-N:]),2)) + " / " + str(round(np.mean(f[-N:]),2)) + " / "  + str(round(np.mean(a[-N:]),2)) + " / " + str(round(np.mean(c[-N:]),2))

# plt.figure(1)

# plt.subplot(411)
# plt.tight_layout()
# plt.plot(np.linspace(0, 1000 * len(a), len(a)), a)
# plt.title('Testing Reward')
# # plt.xlabel('Training Steps')
# #plt.ylabel('Average Reward, (w=' + str(window) + ')')
# # plt.xlim([0,10000000])

# plt.subplot(412)
# plt.tight_layout()
# plt.plot(np.linspace(0, len(b)* 1000, len(b)), b)
# plt.title('Model Order')
# # plt.xlabel('Training Episodes')
# #plt.ylabel('Episode Length (steps),(w=' + str(window2) + ')')

# plt.subplot(413)
# plt.tight_layout()
# plt.plot(np.linspace(0, len(c) * 1000, len(c)), c)
# plt.title('Testing Error')
# # plt.xlabel('Training Steps')
# #plt.ylabel('Model Order')

# plt.subplot(414)
# plt.tight_layout()
# plt.plot(np.linspace(0, len(d) * 1000, len(d)), d)
# plt.title('Training Error')
# # plt.xlabel('Training Steps')
# #plt.ylabel('Training Error')
# plt.tight_layout()
# plt.show()



