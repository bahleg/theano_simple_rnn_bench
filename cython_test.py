import numpy as np
import sys
import time
from naive_cython import full_encode

params = np.load('params.npy')
W = params


	
if len(sys.argv)>1 and sys.argv[1]=='-v':
	data = np.load('data_v.npy')
	print 'validation'
	print full_encode(data, W)

else:
	data = np.load('data.npy')
	print 'running'
	results = []
	for _ in xrange(0, 5):
		time_s = time.time() #too lazy to use timeit
		full_encode(data, W)
		results.append(time.time() - time_s)
		print results[-1]
	print np.mean(results), np.std(results)



