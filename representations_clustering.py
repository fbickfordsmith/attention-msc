'''
Cluster the mean VGG16 representations of ImageNet classes.
(Group ImageNet classes by the similarity of their mean VGG16 representation.)
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation, AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from clustering import test_clustering

def test_clustering(algorithm, X, bigcluster_size=50):
    clustering = algorithm.fit(X)
    num_clusters = len(np.unique(clustering.labels_))
    cluster_sizes = np.array([
        np.count_nonzero(clustering.labels_==cluster_ind)
        for cluster_ind in range(num_clusters)])
    bigcluster_inds = np.flatnonzero(cluster_sizes>=bigcluster_size)
    num_bigclusters = len(bigcluster_inds)
    bigcluster_names = []
    for cluster_ind in bigcluster_inds:
        incluster_inds = np.flatnonzero(clustering.labels_==cluster_ind)
        incluster_vecs = X[incluster_inds]
        if 'cluster_centers_' in dir(clustering):
            incluster_dists = euclidean_distances(
                incluster_vecs, clustering.cluster_centers_[cluster_ind][None, :])
        else:
            incluster_dists = euclidean_distances(
                incluster_vecs, np.mean(incluster_vecs, axis=0)[None, :]) # need to check
        mostcentral_inds = incluster_inds[np.argsort(incluster_dists.flatten()[:bigcluster_size])]
        bigcluster_names.append([ind2name[i] for i in mostcentral_inds])
    return cluster_sizes, bigcluster_names, num_bigclusters

path = '~/attention/'
X = np.load(path+'npy/mean_activations.npy')
df = pd.read_csv(path+'csv/baseline_classwise.csv', index_col=0)
ind2name = {ind:name for ind, name in enumerate(df['name'])}
n_init = int(sys.argv[1]) # sys.argv[0] is the name of the script

algorithms = {
    'kmeans': KMeans(n_clusters=10, init='random', n_init=n_init),
    'spec_nn': SpectralClustering(
        n_clusters=10, affinity='nearest_neighbors', n_neighbors=50, n_init=n_init),
    'spec_cos': SpectralClustering(n_clusters=10, affinity='cosine', n_init=n_init),
    'affprop': AffinityPropagation(preference=-4e4), # reasonable results with preference in [-1e4, -1e5]
    'agglom': AgglomerativeClustering(n_clusters=11)} # tried linkage='complete', affinity='cosine'

for alg in algorithms.keys():
    sizes, names, num_bigclusters = test_clustering(algorithms[alg])
    print(f'{alg}: {num_bigclusters} big clusters')
    pd.DataFrame(names).to_csv(
        path+f'csv/clusters_{alg}.csv', header=False, index=False)
