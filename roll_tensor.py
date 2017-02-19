import numpy as np
import theano
import theano.tensor as T
import sys
import time
params = np.load('params.npy')
W = theano.shared(params)
Xs = T.tensor3('X')
def step(i, hidden, X_tensor):
	X_matrix = X_tensor[:,i,:]
	return T.tanh(T.dot(X_matrix, W) + T.dot(hidden, W)) 

print 'compiling'
hidden_0 = Xs[:,0,:]*0

hiddens = [hidden_0]
for i in xrange(0, 10):
	hiddens.append(step(i, hiddens[-1], Xs))


func = theano.function([Xs],  hiddens[-1])

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



