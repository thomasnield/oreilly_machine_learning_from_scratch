import numpy as np
import pandas as pd


# Cluster points using k-means algorithm with average-based heuristic
# Desmos graph: https://www.desmos.com/calculator/pb4ewmqdvy

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "{0},{1}".format(self.x, self.y)


def distance_between(point1, point2):
    return ((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2) ** .5


def closest_centroid_for(point):
    for c in centroids:
        if distance_between(point, c) == min([distance_between(point, c2) for c2 in centroids]):
            return c


def points_for_centroid(centroid):
    for p in points:
        if closest_centroid_for(p) == centroid:
            yield p


# There will be 4 centroids
k = 4

# Declare Point objects
points = [(Point(row.x, row.y)) for index, row in pd.read_csv("https://tinyurl.com/y25lvxug").iterrows()]

# Declare centroid objects
centroids = [Point(np.random.uniform(0,10), np.random.uniform(0,10)) for i in range(k)]

# Move centroids for 1000 iterations using average technique
best_loss = 1_000_000_000.0

for i in range(1_000):

    for c in centroids:

        x_sum = 0.0
        y_sum = 0.0

        clustered_points = list(points_for_centroid(c))

        for p in clustered_points:
            x_sum += p.x
            y_sum += p.y

        if len(clustered_points) > 0:
            c.x = x_sum / len(clustered_points)
            c.y = y_sum / len(clustered_points)

# Print centroids
for c in centroids:
    print("CENTROID: {0}".format(c))

    for p in points_for_centroid(c):
        print("    {0}".format(p))
