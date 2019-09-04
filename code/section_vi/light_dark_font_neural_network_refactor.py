import numpy as np
import pandas as pd
from scipy import special

training_data = pd.read_csv("https://tinyurl.com/y2qmhfsr")
training_data_count = len(training_data.index)

learning_rate = .1

# Extract the input columns
# We divide the RGB colors by 255 to scale them down between 0 and 1
training_inputs = training_data.iloc[:, 0:3].values.transpose() / 255

# Extract output column, and generate an opposite column where 1 is 0 and 0 is 1.
actual_outputs = np.vstack(
    (training_data.iloc[:, -1].values.transpose(), -1 * (training_data.iloc[:, -1].values.transpose() - 1)))

# Build neural network with weights and biases, starting with random values
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

best_loss = 10_000_000_000


def randomly_adjust_weights(matrix):
    adjust = np.random.normal() * learning_rate

    row = np.random.randint(0, matrix.shape[0])
    col = np.random.randint(0, matrix.shape[1])

    if matrix[row, col] + adjust < -1.0:
        adjust = -1.0 - matrix[row, col]
    if matrix[row, col] + adjust > 1.0:
        adjust = 1.0 - matrix[row, col]

    matrix[row, col] += adjust

    return row, col, adjust


def randomly_adjust_biases(matrix):
    adjust = np.random.normal() * learning_rate

    row = np.random.randint(0, matrix.shape[0])
    col = 0

    if matrix[row, col] + adjust < 0.0:
        adjust = 0.0 - matrix[row, col]
    if matrix[row, col] + adjust > 1.0:
        adjust = 1.0 - matrix[row, col]

    matrix[row, col] += adjust

    return row, col, adjust


for i in range(1_000_000):

    # 32 hyperparameters to randomly select from
    random_select = np.random.randint(0, 32)
    selected_matrix = None

    if random_select < 9:
        selected_matrix = input_weights
    elif random_select < 18:
        selected_matrix = middle_weights
    elif random_select < 24:
        selected_matrix = output_weights
    elif random_select < 27:
        selected_matrix = input_bias
    elif random_select < 30:
        selected_matrix = middle_bias
    elif random_select < 32:
        selected_matrix = output_bias

    random_row, random_col, random_adjust = None, None, None

    # Randomly adjust a weight matrix or a bias matrix
    if random_select < 24:
        random_row, random_col, random_adjust = randomly_adjust_weights(selected_matrix)
    else:
        random_row, random_col, random_adjust = randomly_adjust_biases(selected_matrix)

    training_outputs = softmax(output_bias + output_weights.dot(tanh(middle_bias + middle_weights.dot(relu(input_bias + input_weights.dot(training_inputs))))))
    mean_loss = np.sum((actual_outputs - training_outputs) ** 2) / training_data_count

    # If there's an improvement in loss, keep this new weight/bias adjustment
    if mean_loss < best_loss:
        best_loss = mean_loss
        print(best_loss)
    else:
        # If there isn't an improvement, undo the adjustment
        selected_matrix[random_row, random_col] -= random_adjust


# Interact and test with new colors
def predict_probability(r, g, b):
    input_colors = np.array([r, g, b]).transpose() / 255
    output = softmax(output_bias + output_weights.dot(
        tanh(middle_bias + middle_weights.dot(relu(input_bias + input_weights.dot(input_colors))))))
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
