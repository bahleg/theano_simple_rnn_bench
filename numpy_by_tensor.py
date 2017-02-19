import numpy as np
import sys
import time
params = np.load('params.npy')
W = params

def func(X):
	hidden = np.zeros((X.shape[0],X.shape[2]))
	for i in xrange(0, 10):
		X_matrix = X[:,i,:]
		hidden = np.tanh(np.dot(X_matrix ,W)+np.dot(hidden, W))
	return hidden
	
if len(sys.argv)>1 and sys.argv[1]=='-v':
	data = np.load('data_v.npy')
	print 'validation'
	
	print func(data)
else:
	data = np.load('data.npy')
	print 'running'
	results = []
	for _ in xrange(0, 5):
		time_s = time.time() #too lazy to use timeit
		func(data)
		results.append(time.time() - time_s)
		print results[-1]
	print np.mean(results), np.std(results)
