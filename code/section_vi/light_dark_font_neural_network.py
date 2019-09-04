import numpy as np
import pandas as pd
from scipy import special

training_colors = pd.read_csv("https://tinyurl.com/y2qmhfsr")
training_colors_count = len(training_colors.index)

# Extract the input columns
input_colors = training_colors.iloc[:, 0:3].values.transpose()

# Extract output column, and generate an opposite column where 1 is 0 and 0 is 1.
actual_outputs = np.vstack(
    (training_colors.iloc[:, -1].values.transpose(), -1 * (training_colors.iloc[:, -1].values.transpose() - 1)))

# Build neural network
input_weights = np.random.rand(3, 3)
middle_weights = np.random.rand(3, 3)
output_weights = np.random.rand(2, 3)

input_bias = np.random.rand(3, 1)
middle_bias = np.random.rand(3, 1)
output_bias = np.random.rand(2, 1)


# Activation functions
def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    return special.softmax(x, axis=0)


# Execute training with hill-climbing
for i in range(10000):

    # 32 hyperparameters to randomly select from

    random_select = np.random.randint(0, 32)
    random_adjust = np.random.normal()

    if random_select < 9:
        input_weights[random_select] += random_adjust
    elif random_select < 18:
        middle_weights[random_select - 9] += random_adjust
    elif random_select < 24:
        middle_weights[random_select - 18] += random_adjust


    training_outputs = softmax(output_bias + output_weights.dot(relu(middle_bias + middle_weights.dot(input_bias + input_weights.dot(input_colors)))))

    mean_loss = np.sum((actual_outputs - training_outputs) ** 2) / training_colors_count

