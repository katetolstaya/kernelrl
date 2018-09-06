import numpy as np
import scipy.sparse.linalg
import pickle
from algs.knaf import KNAFModel
from itertools import combinations
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser

def softmax(a,b):
	return np.exp(a) / (np.exp(a) + np.exp(b))

def scores(models, x, ind):

	scores = np.zeros((len(ind),))
	j = 0
	for i in ind:
		scores[j] = np.sum(models[i].vpl.kernel.f(x, models[i].vpl.D), axis=1)
		j = j+1

	return ind[np.argmax(scores)]

# models to combine
#model_names = [ "robot_results/rob14_model54.txt", "robot_results/rob15_model25.txt", "exp18/rob16_model74.txt"] 

model_names = ["robot_results/exp20/rob20_model23.txt", "robot_results/exp21/rob21_model63.txt",  "robot_results/exp23/rob23_3_model3.txt", "robot_results/exp22/rob22_model50.txt"]

# load models
models = list()
n_centers = 0
for s in model_names:
	m = pickle.load(open(s,'rb'))
	models.append(m)
	n_centers = n_centers + np.shape(m.vpl.D)[0]
	print np.shape(m.vpl.D)[0]

# assemble all dictionary points
centers = np.zeros((n_centers,5))
c_model = np.zeros((n_centers,),dtype=int)
ys = np.zeros((n_centers,3))
mi = 0
for i in range(0,len(models)):
	l = np.shape(models[i].vpl.D)[0]
	centers[np.arange(mi, mi+l),:] = models[i].vpl.D
	c_model[np.arange(mi, mi+l)] = i
	ys[np.arange(mi, mi+l),:] = models[i].vpl(models[i].vpl.D)
	mi=mi+l

# set up new model
fname = 'cfg/knaf_robot.cfg'
config = ConfigParser()
with open(fname, 'r') as f:
    config.read_file(f)
model = KNAFModel(5, 1, config[config.default_section])

KDD = np.zeros((n_centers, n_centers))

for i in range(0, n_centers):
	KDD[i,:] = model.vpl.kernel.f(centers[i,:], centers)

weights = np.linalg.lstsq(KDD,ys)[0]

model.vpl.append(centers,weights)
print np.shape(model.vpl.D)

model.vpl.prune(model.eps)

print np.shape(model.vpl.D)

pickle.dump(model,open( "lst.txt",'wb'))

