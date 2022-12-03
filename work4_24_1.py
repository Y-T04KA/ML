import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans

def evk(point, data):
    return np.sqrt(np.sum((point - data) ** 2, axis=1))

class MyKmeans:
    def __init__(self, n_clusters=8, max_iter=300, eps=0.01):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.eps = eps

    def fit(self, data, M):
        self.centroids = M
        iteration = 0
        test = False
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter and test == False:
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in data:
                dists = evk(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
                if (self.centroids[i][0] - prev_centroids[i][0] < self.eps) or (
                        self.centroids[i][1] - prev_centroids[i][1]):
                    test = True
                    break
            iteration += 1

    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = evk(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idxs


def plotnik(d, kmeans):
    class_centers, classification = kmeans.evaluate(d)
    sns.scatterplot(x=[X[0] for X in d],
                    y=[X[1] for X in d],
                    hue=classification,
                    palette="deep",
                    legend=None
                    )
    plt.plot([x for x, _ in kmeans.centroids],
             [y for _, y in kmeans.centroids],
             'k+',
             markersize=10,
             )
    plt.show()


def plot_sk(d, kmeans):
    label = kmeans.fit_predict(d)
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(d[label == i, 0], d[label == i, 1], label=i)
    centroids = kmeans.cluster_centers_
    for i in u_labels:
        plt.scatter(d[label == i, 0], d[label == i, 1], label=i)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
    plt.legend()
    plt.show()


d = np.array([[-40, -10.1], [-20.6, -19.6], [-13.7, -19.4], [-34.6, -10.2], [-34.6, -11.6], [-39.9, -11], [-37, -12],
              [-22.6, -20.6], [-20.3, -16.2], [-15.9, -19.3]])
k = 2
M = np.array([[-41, -13],
              [-19, -14]])

kmeans1 = KMeans(n_clusters=k, random_state=0).fit(d)
plot_sk(d, kmeans1)
kmeans = MyKmeans(n_clusters=k)
kmeans.fit(d, M)
plotnik(d, kmeans)
#####################
d = np.array([[-40, -10.1], [-20.6, -19.6], [-13.7, -19.4], [-34.6, -10.2], [-34.6, -11.6], [-39.9, -11], [-37, -12],
              [-22.6, -20.6], [-20.3, -16.2], [-15.9, -19.3], [-8, -10]])
kmeans1 = KMeans(n_clusters=k, random_state=0).fit(d)
plot_sk(d, kmeans1)
kmeans = MyKmeans(n_clusters=k)
kmeans.fit(d, M)
plotnik(d, kmeans)
