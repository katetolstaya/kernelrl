import numpy as np
import pickle
from algs.knaf import KNAFModel
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser

def softmax(a,b):
	return np.exp(a) / (np.exp(a) + np.exp(b))

# models to combine
model_names = [ "robot_results/rob14_model54.txt", "robot_results/rob15_model25.txt", "exp18/rob16_model74.txt"] 

# load models
models = list()
n_centers = 0
for s in model_names:
	m = pickle.load(open(s,'rb'))
	models.append(m)
	n_centers = n_centers + np.shape(m.vpl.D)[0]

# assemble all dictionary points
centers = np.zeros((n_centers,5))
c_model = np.zeros((n_centers,),dtype=int)
mi = 0
for i in range(0,len(models)):
	l = np.shape(models[i].vpl.D)[0]
	centers[np.arange(mi, mi+l),:] = models[i].vpl.D
	c_model[np.arange(mi, mi+l)] = i
	mi=mi+l

# set up new model
fname = 'cfg/knaf_robot.cfg'
config = ConfigParser()
with open(fname, 'r') as f:
    config.read_file(f)
model = KNAFModel(5, 1, config[config.default_section])

# analyze points in random order
for i in np.random.permutation(np.shape(centers)[0]): #range(0,np.shape(centers)[0]):

	# get center i and its model
	x = centers[i,:]
	model_ = models[c_model[i]]

	# density scores at x
	score_ = np.sum(model_.vpl.kernel.f(x, model_.vpl.D), axis=1)
	score = np.sum(model.vpl.kernel.f(x, model.vpl.D), axis=1)

	# update model
	if score_ > score:
		model.vpl.append(x, (model_.vpl(x) - model.vpl(x)))

model.vpl.prune(model.eps)

# save model
print (np.shape(model.vpl.D))
pickle.dump(model,open("comod3.txt",'wb'))





