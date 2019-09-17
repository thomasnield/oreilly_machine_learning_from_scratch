import numpy as np
import pandas as pd
from scipy import special

training_data = pd.read_csv("https://tinyurl.com/y2qmhfsr")
training_data_count = len(training_data.index)

# Learning rate controls how slowly we approach a solution
# Make it too small, it will take too long to run.
# Make it too big, it will likely overshoot and miss the solution.
learning_rate = 0.1

# Extract the input columns, scale down by 255
training_inputs = training_data.iloc[:, 0:3].values.transpose() / 255

# Extract output column, and generate an opposite column where 1 is 0 and 0 is 1.
actual_outputs = np.vstack(
    (training_data.iloc[:, -1].values.transpose(), -1 * (training_data.iloc[:, -1].values.transpose() - 1)))

# Build a 4-layer neural network with weights and biases
# Input layer does not get weights

middle_layer1_weights = np.random.rand(3, 3)
middle_layer2_weights = np.random.rand(3, 3)
output_weights = np.random.rand(2, 3)

middle_layer1_biases = np.random.rand(3, 1)
middle_layer2_biases = np.random.rand(3, 1)
output_bias = np.random.rand(2, 1)


# Activation functions
def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    return special.softmax(x, axis=0)

best_loss = 10_000_000_000

# Execute training with hill-climbing
for i in range(1_000_000):

    # 32 hyperparameters to randomly select from
    random_select = np.random.randint(0, 32)
    random_adjust = np.random.normal() * learning_rate
    random_row = 0
    random_col = 0

    if random_select < 9:
        random_row = np.random.randint(0, 3)
        random_col = np.random.randint(0, 3)

        if middle_layer1_weights[random_row, random_col] + random_adjust < -1.0:
            random_adjust = -1.0 - middle_layer1_weights[random_row, random_col]
        if middle_layer1_weights[random_row, random_col] + random_adjust > 1.0:
            random_adjust = 1.0 - middle_layer1_weights[random_row, random_col]

        middle_layer1_weights[random_row, random_col] += random_adjust

    elif random_select < 18:
        random_row = np.random.randint(0, 3)
        random_col = np.random.randint(0, 3)

        if middle_layer1_weights[random_row, random_col] + random_adjust < -1.0:
            random_adjust = -1.0 - middle_layer2_weights[random_row, random_col]
        if middle_layer1_weights[random_row, random_col] + random_adjust > 1.0:
            random_adjust = 1.0 - middle_layer2_weights[random_row, random_col]

        middle_layer2_weights[random_row, random_col] += random_adjust

    elif random_select < 24:
        random_row = np.random.randint(2)
        random_col = np.random.randint(3)

        if middle_layer1_weights[random_row, random_col] + random_adjust < -1.0:
            random_adjust = -1.0 - output_weights[random_row, random_col]
        if middle_layer1_weights[random_row, random_col] + random_adjust > 1.0:
            random_adjust = 1.0 - output_weights[random_row, random_col]

        output_weights[random_row, random_col] += random_adjust

    elif random_select < 27:
        random_row = np.random.randint(3)
        random_col = 0

        if middle_layer1_weights[random_row, random_col] + random_adjust < 0.0:
            random_adjust = 0.0 - middle_layer1_biases[random_row, random_col]
        if middle_layer1_weights[random_row, random_col] + random_adjust > 1.0:
            random_adjust = 1.0 - middle_layer1_biases[random_row, random_col]

        middle_layer1_biases[random_row, random_col] += random_adjust

    elif random_select < 30:
        random_row = np.random.randint(3)
        random_col = 0

        if middle_layer1_weights[random_row, random_col] + random_adjust < 0.0:
            random_adjust = 0.0 - middle_layer2_biases[random_row, random_col]
        if middle_layer1_weights[random_row, random_col] + random_adjust > 1.0:
            random_adjust = 1.0 - middle_layer2_biases[random_row, random_col]

        middle_layer2_biases[random_row, random_col] += random_adjust

    elif random_select < 32:
        random_row = np.random.randint(2)
        random_col = 0

        if middle_layer1_weights[random_row, random_col] + random_adjust < 0.0:
            random_adjust = 0.0 - output_bias[random_row, random_col]
        if middle_layer1_weights[random_row, random_col] + random_adjust > 1.0:
            random_adjust = 1.0 - output_bias[random_row, random_col]

        output_bias[random_row, random_col] += random_adjust

    # Calculate outputs with the given weights, biases, and activation functions for all three layers
    training_outputs = softmax(output_bias + output_weights.dot(tanh(middle_layer2_biases + middle_layer2_weights.dot(relu(middle_layer1_biases + middle_layer1_weights.dot(training_inputs))))))

    # Calculate the mean squared loss
    mean_loss = np.sum((actual_outputs - training_outputs) ** 2) / training_data_count

    # If the loss improves, keep the random adjustment. Otherwise revert.
    if mean_loss < best_loss:
        best_loss = mean_loss
        print(best_loss)

    elif random_select < 9:
        middle_layer1_weights[random_row, random_col] -= random_adjust

    elif random_select < 18:
        middle_layer2_weights[random_row, random_col] -= random_adjust

    elif random_select < 24:
        output_weights[random_row, random_col] -= random_adjust

    elif random_select < 27:
        middle_layer1_biases[random_row, random_col] -= random_adjust

    elif random_select < 30:
        middle_layer2_biases[random_row, random_col] -= random_adjust

    elif random_select < 32:
        output_bias[random_row, random_col] -= random_adjust


# Interact and test with new colors
def predict_probability(r, g, b):
    input_colors = np.array([r,g,b]).transpose() / 255
    output = softmax(output_bias + output_weights.dot(tanh(middle_layer2_biases + middle_layer2_weights.dot(relu(middle_layer1_biases + middle_layer1_weights.dot(input_colors))))))
    return output

def predict_font_shade(r, g, b):
    output_values = predict_probability(r, g, b)
    if output_values[0,0] > output_values[1,0]:
        return "DARK"
    else:
        return "LIGHT"

while True:
    n = input("Predict light or dark font. Input values R,G,B: ")
    (r, g, b) = n.split(",")
    print(predict_font_shade(int(r), int(g), int(b)))
