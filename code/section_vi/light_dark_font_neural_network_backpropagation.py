# Helpful resource: https://www.kaggle.com/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

all_data = pd.read_csv("https://tinyurl.com/y2qmhfsr")

# Learning rate controls how slowly we approach a solution
# Make it too small, it will take too long to run.
# Make it too big, it will likely overshoot and miss the solution.
L = 0.1
sample_size = 100

# Extract the input columns, scale down by 255
all_inputs = (all_data.iloc[:, 0:3].values / 255.0)
all_outputs = all_data.iloc[:, -1].values

# Split train and test data sets
X_train, X_test, Y_train, Y_test = train_test_split(all_inputs, all_outputs, test_size=1/3)
n = X_train.shape[0]


# Build neural network with weights and biases
# with random initialization
hidden_w = np.random.rand(3, 3)
output_w = np.random.rand(1, 3)

hidden_b = np.random.rand(3, 1)
output_b = np.random.rand(1, 1)

# Activation functions
relu = lambda x: np.maximum(x, 0)
logistic = lambda x: 1 / (1 + np.exp(-x))

# Derivatives of Activation functions
d_relu = lambda x: x > 0
d_logistic = lambda x: np.exp(-x)/(1 + np.exp(-x))**2

# Runs inputs through the neural network to get predicted outputs
def forward_prop(X):
    Z1 = hidden_w @ X + hidden_b
    A1 = relu(Z1)
    Z2 = output_w @ A1 + output_b
    A2 = logistic(Z2)
    return Z1, A1, Z2, A2

# returns slopes for weights and biases
def backward_prop(Z1, A1, Z2, A2, X, Y):
    dZ2 = A2 - Y
    dW2 = 1 / n * dZ2 @ A1.transpose()
    db2 = 1 / n * np.sum(dZ2)
    dZ1 = output_w.transpose() @ dZ2 * d_relu(Z1)
    dW1 = 1 / n * dZ1 @ X.transpose()
    db1 = 1 / n * np.sum(dZ1)
    return dW1, db1, dW2, db2

# Execute gradient descent
for i in range(100_000):

    # randomly select part of the training data
    idx = np.random.choice(n, sample_size, replace=False)
    X_sample = X_train[idx].transpose()
    Y_sample = Y_train[idx]

    # run randomly selected training data through neural network
    Z1, A1, Z2, A2 = forward_prop(X_sample)

    # distribute error through backpropogation
    # and return slopes for weights and biases
    dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, X_sample, Y_sample)

    # update weights and biases
    hidden_w -= L * dW1
    hidden_b -= L * db1
    output_w -= L * dW2
    output_b -= L * db2

# Calculate accuracy
test_predictions = forward_prop(X_test.transpose())[3]
test_comparisons = np.equal((test_predictions >= .5).flatten().astype(int), Y_test)
accuracy = sum(test_comparisons.astype(int) / X_test.shape[0])
print("ACCURACY: ", accuracy)


# Interact and test with new colors
def predict_probability(r, g, b):
    X = np.array([[r, g, b]]).transpose() / 255
    Z1, A1, Z2, A2 = forward_prop(X)
    return A2


def predict_font_shade(r, g, b):
    output_values = predict_probability(r, g, b)
    if output_values > .5:
        return "DARK"
    else:
        return "LIGHT"

while True:
    col_input = input("Predict light or dark font. Input values R,G,B: ")
    (r, g, b) = col_input.split(",")
    print(predict_font_shade(int(r), int(g), int(b)))
