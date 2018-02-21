import numpy as np
import pickle

#model_names = [  "robot_results/rob16_model49.txt", "robot_results/rob14_model53.txt"]  # big one first
model_names = [ "robot_results/rob14_model54.txt", "robot_results/rob15_model25.txt"] #"robot_results/rob16_model49.txt",] #, "robot_results/rob16_model49.txt",   ] # small one first

def softmax(a,b):
	return np.exp(a) / (np.exp(a) + np.exp(b))

f = None

for s in model_names:
	if f is None:
		f = pickle.load(open(s,'rb'))
	else:
		f_ = pickle.load(open(s,'rb'))

		score_ = np.sum(f_.vpl.KDD, axis=0) #/ np.shape(f_.vpl.KDD)[0]  # for each f_i
		score = np.sum(f.vpl.kernel.f(f.vpl.D,f_.vpl.D), axis=0) #/ np.shape(f.vpl.KDD)[0]

		f_xv = f_.vpl(f_.vpl.D) 
		fxv = f.vpl(f_.vpl.D) 

		for j in range(0,np.shape(f_.vpl.D)[0]):
			x = f_.vpl.D[j,:]

			#step = int(np.maximum(score_[j],score[j]) == score_[j]) #- badddd
			#step1 = int(np.maximum(fxv[j,0],f_xv[j,0]) == f_xv[j,0]) # compare V values

			step = softmax(score_[j],score[j]) * softmax(f_xv[j,0],fxv[j,0]) #np.exp(score_[j]) / (np.exp(score[j]) + np.exp(score_[j])) # softmax


			print step

			#print step * (f_xv[j,:] - f.vpl(x))

			f.vpl.append(x, (f_xv[j,:] - f.vpl(x)) * step)

		f.vpl.prune(f.eps)

print np.sum(f.vpl(f_.vpl.D)[:,0] - f_.vpl(f_.vpl.D)[:,0]  )

print (np.shape(f.vpl.D))
pickle.dump(f,open("combined_model11.txt",'wb'))


	# #scoresv = np.zeros((len(models),))
	# scores = np.zeros((len(models),))
	# for j in range(0,len(models)):
	# 	scores[j] = np.sum(model.vpl.kernel.f(x, models[j].vpl.D), axis=1)
	# 	#scoresv[j] = model.vpl(x)[0,0]

	# max_model = np.argmax(scores) #+scoresv)
	# model.vpl.append(x, (models[max_model].vpl(x) - model.vpl(x)))




