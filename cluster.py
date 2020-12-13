from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

def pca_components(waves):

    pca = PCA()
    waves = pca.fit_transform(waves)
    t = sum(pca.explained_variance_)
    n_c = 0 
    variance = 0
    while variance/t < 0.90:
        variance += pca.explained_variance_[n_c]
        n_c += 1
    
    return n_c

def reduce(waves, scaler, n_c):

    scaled = scaler.fit_transform(waves)
    pca = PCA(n_components=n_c)
    reduced = pca.fit_transform(scaled)

    return reduced

def clustering(data, num_clusters, n_iters):

    kmeans = KMeans(init="random", n_clusters=num_clusters, max_iter=n_iters, random_state=42)
    kmeans.fit(data)

    clusters = kmeans.predict(data)
    centers = kmeans.cluster_centers_

    return clusters, centers