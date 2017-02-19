import numpy as np
import sys
import time
from numba import jit
params = np.load('params.npy')
W = params

@jit
def func(X):
	hidden = np.zeros(100)
	for i in xrange(0, 10):
		hidden = np.tanh(np.dot(X[i],W)+np.dot(hidden, W))
	return hidden
	
if len(sys.argv)>1 and sys.argv[1]=='-v':
	data = np.load('data_v.npy')
	print 'validation'
	for x in data:
		print func(x)
else:
	data = np.load('data.npy')
	print 'running'
	results = []
	for _ in xrange(0, 5):
		time_s = time.time() #too lazy to use timeit
		for x in data:
			func(x)
		results.append(time.time() - time_s)
		print results[-1]
	print np.mean(results), np.std(results)



