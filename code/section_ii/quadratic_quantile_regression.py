import pandas as pd
from random import uniform, normalvariate, randint
import numpy as np

# Generate random data for y = 3x^2 + 20
# Desmos graph: https://www.desmos.com/calculator/doesy3llvy

def f(x):
    return 3 * x**2 + 20

data = pd.DataFrame(columns=["x", "y"])

for i in range(1000):
    x = round(uniform(1, 20), 2)
    data = data.append(pd.DataFrame(columns=["x", "y"], data=[[x, round(f(x) + normalvariate(0, 50), 2)]]))

data.to_csv("input_data.csv")

# Input data
X = data.iloc[:, 0]
Y = data.iloc[:, 1]

# Building the model
a = 0.0
b = 0.0
under_curve_target = .80

epochs = 200000  # The number of iterations to perform

n = float(len(X))  # Number of elements in X


best_loss = 10000000000000.0  # Initialize with a really large value
best_percent_under_curve = 0.0

for i in range(epochs):

    # Randomly adjust "a" and "b"

    a_adjust = np.random.standard_normal()
    b_adjust = np.random.standard_normal()

    a += a_adjust
    b += b_adjust

    # Calculate loss, which is total mean squared error
    new_loss = (1 / n) * sum((Y - (a * X ** 2 + b)) ** 2)

    # Calculate percentage of points below line
    percent_under_curve = sum(Y <= (a * X ** 2 + b)) / n

    # Evaluate move
    if best_percent_under_curve < under_curve_target and percent_under_curve > best_percent_under_curve:
        best_percent_under_curve = percent_under_curve
        best_loss = new_loss
    elif best_percent_under_curve >= under_curve_target and percent_under_curve >= under_curve_target and new_loss < best_loss:
        best_loss = new_loss
    else:
        a -= a_adjust
        b -= b_adjust


    if i % 1000 == 0:
        print(a, b)

print(a, b)
