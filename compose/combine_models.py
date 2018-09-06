import numpy as np
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

# # assemble all dictionary points
# centers = np.zeros((n_centers,5))
# c_model = np.zeros((n_centers,),dtype=int)
# mi = 0
# for i in range(0,len(models)):
# 	l = np.shape(models[i].vpl.D)[0]
# 	centers[np.arange(mi, mi+l),:] = models[i].vpl.D
# 	c_model[np.arange(mi, mi+l)] = i
# 	mi=mi+l

# # set up new model
# fname = 'cfg/knaf_robot.cfg'
# config = ConfigParser()
# with open(fname, 'r') as f:
#     config.read_file(f)

# flist = open('combos/list.txt','wb')
# flist.write(str(model_names) + '\n')

# n_save = 0
# for n_c in range(1,len(models)+1):

# 	# for each combo, let's make a model
# 	for ind in combinations(range(0,len(models)),n_c):	
# 		model = KNAFModel(5, 1, config[config.default_section])

# 		# analyze points in random order
# 		for i in np.random.permutation(np.shape(centers)[0]): #range(0,np.shape(centers)[0]):

# 			if c_model[i] in ind: # if this dictionary point is from a policy in the combo
# 				# get center i and its model
# 				x = centers[i,:]
# 				model_ = models[c_model[i]]

# 				# density scores at x
# 				#score_ = np.sum(model_.vpl.kernel.f(x, model_.vpl.D), axis=1)
# 				#score = np.sum(model.vpl.kernel.f(x, model.vpl.D), axis=1)

# 				# update model
# 				if scores(models, x, ind) == c_model[i]: #> score:

# 					model.vpl.append(x, (model_.vpl(x) - model.vpl(x)))

# 		model.vpl.prune(model.eps)

# 		# save model
# 		print (np.shape(model.vpl.D))
# 		pickle.dump(model,open("combos/policy" + str(n_save) + ".txt",'wb'))
# 		flist.write("combos/policy" + str(n_save) + ".txt, " + str(ind) + '\n')
# 		n_save = n_save + 1







