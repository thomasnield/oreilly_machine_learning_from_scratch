import random

import numpy as np
import pandas as pd


# Cluster points using k-means algorithm using hill climbing
# Desmos graph: https://www.desmos.com/calculator/pb4ewmqdvy

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "{0},{1}".format(self.x, self.y)

# Calculate distance between two points
def distance_between(point1, point2):
    return ((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2) ** .5

# Retrieve closest centroid for given point
def closest_centroid_for(point):
    for c in centroids:
        if distance_between(point, c) == min([distance_between(point, c2) for c2 in centroids]):
            return c


# Get points that belong to the given centroid
def points_for_centroid(centroid):
    for p in points:
        if closest_centroid_for(p) == centroid:
            yield p

# We will have 4 centroids
k = 4

# Declare points
points = [(Point(row.x, row.y)) for index, row in pd.read_csv("https://tinyurl.com/y25lvxug").iterrows()]

# Declare centroids
centroids = [Point(0, 0) for i in range(k)]

# Randomly move centroids, keep moves that reduce loss which will converge on a solution
best_loss = 1_000_000_000.0

for i in range(200_000):
    random_centroid = random.choice(centroids)

    random_x_adjust = np.random.standard_normal()
    random_y_adjust = np.random.standard_normal()

    random_centroid.x += random_x_adjust
    random_centroid.y += random_y_adjust

    new_loss = 0.0

    for p in points:
        new_loss += distance_between(p, closest_centroid_for(p))**2

    if new_loss < best_loss:
        best_loss = new_loss
    else:
        random_centroid.x -= random_x_adjust
        random_centroid.y -= random_y_adjust

# Print the centroids and their points
for c in centroids:
    print("CENTROID: {0}".format(c))

    for p in points_for_centroid(c):
        print("    {0}".format(p))



# plot
import matplotlib.pyplot as plt
import numpy as np

# show in chart
X = np.array([p.x for p in points])
Y = np.array([p.y for p in points])
XC = np.array([p.x for p in centroids])
YC = np.array([p.y for p in centroids])
plt.plot(X, Y, 'o') # scatterplot
plt.plot(XC, YC, 'x') # centroids

plt.show()
