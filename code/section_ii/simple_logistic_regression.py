import random

import numpy as np
import pandas as pd
import math


# Desmos graph: https://www.desmos.com/calculator/6cb10atg3l

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "{0},{1}".format(self.x, self.y)


points = [(Point(row.x, row.y)) for index, row in pd.read_csv("https://tinyurl.com/y2cocoo7").iterrows()]

best_likelihood = -10_000_000
b0 = .01
b1 = .01


# calculate maximum likelihood

def predict_probability(x):
    p = 1.0 / (1.0 + math.exp(-(b0 + b1 * x)))
    return p


for i in range(1_000_000):

    # Select b0 or b1 randomly, and adjust it randomly
    random_b = random.choice(range(2))

    random_adjust = np.random.normal()

    if random_b == 0:
        b0 += random_adjust
    elif random_b == 1:
        b1 += random_adjust

    true_estimates = sum(math.log(predict_probability(p.x)) for p in points if p.y == 1.0)
    false_estimates = sum(math.log(1.0 - predict_probability(p.x)) for p in points if p.y == 0.0)

    total_likelihood = true_estimates + false_estimates

    if best_likelihood < total_likelihood:
        best_likelihood = total_likelihood
    elif random_b == 0:
        b0 -= random_adjust
    elif random_b == 1:
        b1 -= random_adjust

print("1.0 / (1 + exp(-({0} + {1}*x))".format(b0, b1))
print("BEST LIKELIHOOD: {0}".format(best_likelihood))
