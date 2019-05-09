from pyhdf.SD import SD, SDC
#from corel.model import *
from corel.rbf_regression import *
import pickle
import random

file_num = '249'

fname = 'data/2018_' + file_num + '_30Z.hdf'
hdf_file = SD(fname, SDC.READ)

#print hdf.datasets()

def train_model(data):
	f = Model(2, 3, GradType.VAR)

	inds = data.nonzero()
	print(np.shape(data))
	print(len(inds[0]))
	n = len(inds[0])
	r = range(0,n)
	random.Random(5).shuffle(r)
	n_samples = 0
	for ind in r[0:int(n/3)]:
		i = inds[0][ind]
		j = inds[1][ind]
		temp = (data[i][j] - offset) * scale 

		sample = (np.array([i,j]), temp)
		(loss, model) =  f.train(sample)

		n_samples = n_samples + 1
		if n_samples % 10000 == 0:
			print n_samples, test_model(f,data), (f.var(np.array([i,j]))), f.mom(np.array([i,j]))

		#print(loss)
		#print(j)
	return f

def test_model(f, data):
	n_samples = 0
	temp_loss = 0

	inds = data.nonzero()
	n = len(inds[0])

	r = range(0,n)
	random.shuffle(r)
	for ind in r[0:int(n/20)]:
		i = inds[0][ind]
		j = inds[1][ind]
		temp = (data[i][j] - offset) * scale 
		sample = (np.array([i,j]), temp)
		temp_loss = temp_loss + f.loss(sample)
		n_samples = n_samples + 1

	return temp_loss/n_samples


dfield_name='sst'
sds_obj = hdf_file.select(dfield_name)
data = sds_obj.get()

attrs = sds_obj.attributes()
offset = attrs.get('add_offset') # subtract this from values to get sst in celcius
scale = attrs.get('scale_factor')

f = train_model(data)
print(test_model(f, data))
pickle.dump(f, open('temps' + file_num + '.pkl','wb'))

#print(attrs)
#print(data.nonzero()) # zero values are NaNs, # convert index to lat, lon




