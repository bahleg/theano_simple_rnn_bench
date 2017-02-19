import numpy as np
import theano
import theano.tensor as T
import sys
import time
params = np.load('params.npy')
W = theano.shared(params)
Xs = theano.shared(np.zeros((100,100,100)))
ind = T.iscalar()
X = Xs[ind, :, :]
def step(i, hidden, X_tensor):
	return T.tanh(T.dot(X_tensor[i], W) + T.dot(hidden, W)) 
print 'compiling'
scan_func = theano.scan(step, sequences=T.arange(0, 10), outputs_info=[np.zeros(100)], non_sequences=X)

func = theano.function([ind], [scan_func[0][-1]])

if len(sys.argv)>1 and sys.argv[1]=='-v':
	data = np.load('data_v.npy')
	Xs.set_value(data)
	print 'validation'
	for i,_ in enumerate(data):
		print func(i)
else:
	data = np.load('data.npy')
	Xs.set_value(data)
	print 'running'
	results = []
	for _ in xrange(0, 5):
		time_s = time.time() #too lazy to use timeit
		for i,_ in enumerate(data):
			func(i)
		results.append(time.time() - time_s)
		print results[-1]
	print np.mean(results), np.std(results)



