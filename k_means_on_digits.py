# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:25:11 2019

@author: Shyam Parmar
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()  # for plot styling
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


digits = load_digits()
digits.data.shape

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape

fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
    


labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]
    

print(accuracy_score(digits.target, labels))

# Confusion matrix
mat = confusion_matrix(digits.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')


# We can use the t-distributed stochastic neighbor embedding (t-SNE) algorithm 
# to pre-process the data before performing k-means. t-SNE is a nonlinear embedding algorithm 
#that is particularly adept at preserving points within clusters. 

'''tsne = TSNE(n_components=2, init='random', random_state=0) # Project the data: this step will take several seconds
digits_proj = tsne.fit_transform(digits.data)

# Compute the clusters
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits_proj)

# Permute the labels
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

# Compute the accuracy
print(accuracy_score(digits.target, labels))'''
