import numpy as np


input_weights = np.zeros([2,3])
input_bias = np.array([1,2,3])

print(input_weights + input_bias)

input_weights[0] += 10000

print(input_weights)
