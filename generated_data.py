import numpy as np
data = np.random.randn(25000,10,100)
data_validate = np.random.randn(2,10,100)
params = np.random.randn(100,100)
np.save('data', data)
np.save('data_v', data_validate)
np.save('params', params)
