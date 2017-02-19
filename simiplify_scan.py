import numpy as np
import theano
import theano.tensor as T
import sys
import time

params = np.load('params.npy')
W = theano.shared(params)
X = T.matrix('X')


def step(i, hidden, X_tensor):
	return T.tanh(X_tensor[i] + T.dot(hidden, W)) 
print 'compiling'

X_dotted = T.dot(X, W)
scan_func = theano.scan(step, sequences=T.arange(0, 10), outputs_info=[np.zeros(100)], non_sequences=X_dotted)

func = theano.function([X], [scan_func[0][-1]])

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



