from coordination import Coordinator
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
import torch
import time

n_samples = 1000
n_dim = 2
true = np.random.random(n_dim * n_samples)
true = true.reshape((n_samples, n_dim))
true -= true.mean()

similarities = euclidean_distances(true)

# Estimate coordinates
t0 = time.time()
coordinator = Coordinator(similarities, dim = n_dim).compile()
coords1 = coordinator.get_coords()
coordinator_time = time.time() - t0

t0 = time.time()
mds = manifold.MDS(n_components=n_dim, dissimilarity="precomputed")
coords2 = mds.fit(similarities).embedding_
mds_time = time.time() - t0

print(f'Coordinator time: {coordinator_time}')
print(f'MDS time: {mds_time}')

# Rescale the data
coords1 *= np.sqrt((true ** 2).sum()) / np.sqrt((coords1 ** 2).sum())
coords2 *= np.sqrt((true ** 2).sum()) / np.sqrt((coords2 ** 2).sum())

# Rotate the data
pca = PCA(n_components=2)
true = pca.fit_transform(true)
coords1 = pca.fit_transform(coords1)
coords2 = pca.fit_transform(coords2)

fig = plt.figure(1)
ax = plt.axes([0., 0., 1., 1.])

plt.scatter(true[:, 0], true[:, 1], color='navy', s=100, label='True',
    marker = '+')
plt.scatter(coords1[:, 0], coords1[:, 1], color='turquoise', s=100, 
    label='Coordinator', marker = 'x')
plt.scatter(coords2[:, 0], coords2[:, 1], color='darkorange', s=100,
    label='MDS', marker = 'x')
plt.legend(scatterpoints=1, loc='best', shadow=False)

plt.show()
