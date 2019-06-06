import pandas as pd
import numpy as np
import random


# Cluster points using k-means algorithm using hill climbing
# This code version uses NumPy
# Desmos graph: https://www.desmos.com/calculator/pb4ewmqdvy

k = 4

points = pd.read_csv("https://tinyurl.com/y25lvxug").values
centroids = np.zeros(k*2, dtype='float').reshape([4,2])

best_loss = 1_000_000_000.0

for i in range(100_000):
    random_centroid = random.choice(centroids)

    random_xy_adjust = np.zeros(k*2).reshape([4,2])
    random_xy_adjust[random.randint(0,k-1)] = np.random.standard_normal(2)
    centroids += random_xy_adjust

    new_loss = 0.0

    for p in points:
        new_loss += min(((p[0] - c[0])**2 + (p[1] - c[1])**2) ** .5 for c in centroids)**2

    if new_loss < best_loss:
        best_loss = new_loss
    else:
        centroids -= random_xy_adjust

for c in centroids:
    print("CENTROID: {0}".format(c))
