import numpy as np
import theano
import theano.tensor as T
import sys
import time
params = np.load('params.npy')
W = theano.shared(params, name='W')
X = T.matrix('X_matrix')
def step(i, hidden, X_tensor):
	return T.tanh(T.dot(X_tensor[i], W) + T.dot(hidden, W)) 

print 'compiling'
hidden_0 = theano.shared(np.zeros(100), name='h0')
hiddens = [hidden_0]
for i in xrange(0, 10):
	hiddens.append(step(i, hiddens[-1], X))


#theano.printing.pydotprint(hiddens[-1], outfile="baseline.png", var_with_name_simple=True) 
#exit(1)

func = theano.function([X],  hiddens[-1])

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



