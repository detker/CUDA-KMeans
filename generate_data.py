from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd


K = 5

data = load_iris()
X = data.data
print(X.shape)
N = X.shape[0]
print(N)
D = X.shape[1]
print(D)

with open('data_custom.txt', 'w') as f:
    f.write(f'{N} {D} {K}\n')
    for i in range(N):
        f.write(' '.join(map(str, X[i])) + '\n')

kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(X)
print(kmeans.cluster_centers_)
print(kmeans.cluster_centers_[0].dtype)
print(kmeans.labels_)

with open('centers_custom.txt', 'w') as f:
    for k in range(K):
        f.write(' '.join(map(str, kmeans.cluster_centers_[k])) + '\n')
    for i in range(N):
        f.write(f'{kmeans.labels_[i]}\n')