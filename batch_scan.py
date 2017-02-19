import numpy as np
import theano
import theano.tensor as T
import sys
import time
params = np.load('params.npy')
W = theano.shared(params)
X = T.tensor3('Xs')
def step(i, hidden, X_tensor):
	return T.tanh(T.dot(X_tensor[i], W) + T.dot(hidden, W)) 
def encode_all(i, X):
	return  theano.scan(step, sequences=T.arange(0, 10), outputs_info=[np.zeros(100)], non_sequences=X[i])[0][-1]
print 'compiling'

scan_func = theano.scan(encode_all, sequences=T.arange(X.shape[0]), non_sequences=X)

func = theano.function([X], [scan_func[0]])

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



