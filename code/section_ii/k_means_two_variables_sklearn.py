import pandas as pd
from sklearn.cluster import KMeans

# Cluster points using SciKit-Learn

# Learn more:
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

# Declare point array
points = pd.read_csv("https://tinyurl.com/y25lvxug")

# Cluster points
kmeans = KMeans(n_clusters=4).fit(points)

# Print the group index of each point
print("\r\nGroupings")
print(kmeans.labels_)

# Print the centroids
print("\r\nCentroids:")
print(kmeans.cluster_centers_)

# Predict a new point (14,5)
print("\r\nPrediction for (14,5)")
print(kmeans.predict([[14,5]]))
