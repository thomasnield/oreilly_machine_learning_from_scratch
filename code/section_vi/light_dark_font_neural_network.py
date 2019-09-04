import numpy as np
import pandas as pd
from scipy import special


training_colors = pd.read_csv("https://tinyurl.com/y2qmhfsr")

# Extract the input columns
input_colors = training_colors.iloc[:,0:3].values.transpose()

# Extract output column
output_actuals = np.vstack((training_colors.iloc[:, -1].values.transpose(), -1 * (training_colors.iloc[:, -1].values.transpose() - 1)))

# Build neural network
input_weights = np.random.rand(3, 3)
middle_weights = np.random.rand(3, 3)
output_weights = np.random.rand(2, 3)


def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    return special.softmax(x, axis=0)

for i in range(10000):
    training_output = softmax(output_weights.dot(relu(middle_weights.dot(input_weights.dot(input_colors)))))

    diff = output_actuals - training_output

    print(diff)

    print("")

