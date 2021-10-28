import numpy as np
import pandas as pd


training_data = pd.read_csv("https://tinyurl.com/y2qmhfsr")
n = len(training_data.index)

# Learning rate controls how slowly we approach a solution
# Make it too small, it will take too long to run.
# Make it too big, it will likely overshoot and miss the solution.
L = 0.01
sample_size = 5

# Extract the input columns, scale down by 255
training_inputs = (training_data.iloc[:, 0:3].values / 255.0)
training_outputs = training_data.iloc[:, -1].values

# Build neural network with weights and biases
middle_weights = np.random.rand(3, 3)
output_weights = np.random.rand(1, 3)

middle_bias = np.random.rand(3, 1)
output_bias = np.random.rand(1, 1)

# Activation functions
softplus = lambda x: np.log(1 + np.exp(x))
logistic = lambda x: 1 / (1 + np.exp(-x))

# Derivatives of Activation functions
d_softplus = lambda x:  np.exp(x)/(np.exp(x) + 1)
d_logistic = lambda x: np.exp(-x)/(1 + np.exp(-x))**2

# Stochastic Gradient descent
for i in range(1_000_000):

    idx = np.random.choice(n, sample_size, replace=False)
    input_sample = training_inputs[idx].transpose()
    output_sample = training_outputs[idx]

    predicted_output = logistic(output_bias + output_weights.dot(softplus(middle_bias + middle_weights.dot(input_sample))))

    error = (predicted_output - output_sample)**2

    # backpropogate?
    print(error)

# Interact and test with new colors
def predict_probability(r, g, b):
    input_colors = np.array([r, g, b]).transpose() / 255
    output = logistic(output_bias + output_weights.dot(softplus(middle_bias + middle_weights.dot(input_colors))))
    return output


def predict_font_shade(r, g, b):
    output_values = predict_probability(r, g, b)
    if output_values[0, 0] > output_values[1, 0]:
        return "DARK"
    else:
        return "LIGHT"


while True:
    n = input("Predict light or dark font. Input values R,G,B: ")
    (r, g, b) = n.split(",")
    print(predict_font_shade(int(r), int(g), int(b)))
