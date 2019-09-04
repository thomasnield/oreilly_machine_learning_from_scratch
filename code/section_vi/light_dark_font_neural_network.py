import numpy as np

training_colors = np.recfromcsv("https://tinyurl.com/y2qmhfsr")

# Build neural network

input_layer = np.random.rand(3,1)
middle_layer = np.random.rand(3,1)
output_layer = np.random.rand(2,1)


print(input_layer)
