
import math

import numpy as np
import pandas as pd
from scipy import special

training_data = pd.read_csv("https://tinyurl.com/y2qmhfsr")


input_weights = np.random.rand(3, 3)

first_output = input_weights.dot(training_inputs)
print(input_layer_weights.dot(training_inputs))


# Activation functions
def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    return special.softmax(x, axis=0)

