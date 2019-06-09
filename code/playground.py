import numpy as np

centroids = np.array([
    [10, 10],
    [0,0],
    [20,20]
])

points = np.array([
    [1,1],
    [2,2],
    [7,7],
    [8,8],
    [17,17],
    [19,19]
])

distances = np.sqrt(((points - centroids[:,np.newaxis])**2).sum(axis=2))

min_distances_indices = np.argmin(distances,axis=0)
min_distances = distances * np.argmin(distances, axis=0)
new_loss = (min_distances ** 2).sum(axis=0)


print(distances)
print("\r\n\r\n")
print(min_distances)