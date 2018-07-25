import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import linalg

#clustering into 3 groups
"""
data1 = loadmat('ex7data2.mat')
X1 = data1['X']
km1 = KMeans(3)
km1.fit(X1)

plt.scatter(X1[:,0], X1[:,1], s=40, c=km1.labels_, cmap=plt.cm.prism) #cmap maps numbers to colors
plt.title('K-Means Clustering Results with K=3')
plt.scatter(km1.cluster_centers_[:,0], km1.cluster_centers_[:,1], marker='+', s=100, c='k', linewidth=2);
plt.show()
"""
#remove """ to implement

#image compression with K-Means

img = plt.imread('bird_small.png')
img_shape = img.shape
A = img/255
AA = A.reshape(128*128,3)
km2 = KMeans(16)
km2.fit(AA)

B = km2.cluster_centers_[km2.labels_].reshape(img_shape[0], img_shape[1], 3)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13,9))
ax1.imshow(img)
ax1.set_title('Original')
ax2.imshow(B*255)
ax2.set_title('Compressed, with 16 colors')
plt.show()
#for ax in fig.axes:
#    ax.axis('off')
#python kmeans.py

