import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle
import cv2

# from PIL import Image

# Generate sample data
centers = [[1, 1], [-.75, -1], [1, -1], [-3, 2]]
X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

#Compute clustering with MeanShift

# The bandwidth can be automatically estimated
bandwidth = estimate_bandwidth(X, quantile=.1,n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

n_clusters_ = labels.max() + 1

#Plot result
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1],
             'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

# Part 2: Color image segmentation using mean shift
image = cv2.imread("C:\\Users\\mohamed ismail\\Desktop\\toy.jpg")[:,:,::-1]

# Need to convert image into feature array based
# on rgb intensities
initial_shape = image.shape
flat_image = np.reshape(image, [-1, 3])
print (flat_image.shape)

# Estimate bandwidth
bandwidth2 = estimate_bandwidth(flat_image,
                                quantile=.2, n_samples=500)
ms = MeanShift(bandwidth2, bin_seeding=True)
ms.fit(flat_image)
labels = ms.labels_

print (labels.shape)
# Plot image vs segmented image
plt.figure(2)
plt.subplot(2, 1, 1)
plt.imshow(image)
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(np.reshape(labels, initial_shape[0:2]))
plt.axis('off')
plt.show()
