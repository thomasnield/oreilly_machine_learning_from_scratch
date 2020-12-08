import random

import numpy as np
import pandas as pd


class Point:
    def __init__(self, x1, x2, y):
        self.x1 = x1
        self.x2 = x2
        self.y = y

    def __str__(self):
        return "{0},{1},{2}".format(self.x1, self.x2, self.y)


points = [(Point(row.x1, row.x2, row.y)) for index, row in pd.read_csv("https://bit.ly/2X1HWH7").iterrows()]

# Building the model y = beta1 + beta2*x1 + beta3*x2
b0 = 0.0
b1 = 0.0
b2 = 0.0

epochs = 100_000  # The number of iterations to perform

n = float(len(points))  # Number of points

best_loss = 10000000000000.0  # Initialize with a really large value

for i in range(epochs):

    # Randomly adjust b0, b1, or b2

    random_b = random.choice(range(3))

    random_adjust = np.random.normal()

    if random_b == 0:
        b0 += random_adjust
    elif random_b == 1:
        b1 += random_adjust
    elif random_b == 2:
        b2 += random_adjust

    # Calculate loss, which is total sum squared error
    new_loss = 0.0
    for p in points:
        new_loss += (p.y - (b0 + b1 * p.x1 + b2 * p.x2)) ** 2

    # If loss has improved, keep new values. Otherwise revert.
    if new_loss < best_loss:
        print("z = {0} + {1}x + {2}y".format(b0, b1, b2))
        best_loss = new_loss
    else:
        if random_b == 0:
            b0 -= random_adjust
        elif random_b == 1:
            b1 -= random_adjust
        elif random_b == 2:
            b2 -= random_adjust

print("z = {0} + {1}x + {2}y".format(b0, b1, b2))
