import math
import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt


def n_counter(ci, cj):
    n11 = 0
    n10 = 0
    n01 = 0
    n00 = 0
    for vi in range(len(ci)):
        if (ci[vi] == cj[vi]) and (ci[vi] == 1):
            n11 += 1
        elif (ci[vi] == cj[vi]) and (ci[vi] == 0):
            n00 += 1
        elif ci[vi] > cj[vi]:
            n10 += 1
        elif ci[vi] < cj[vi]:
            n01 += 1
    return n11, n10, n01, n00


def distanceSMC(p, q):
    n11, n10, n01, n00 = n_counter(p, q)
    smc = (n11 + n00) / (n11 + n10 + n01 + n00)
    return smc


def distanceJC(p, q):
    n11, n10, n01, n00 = n_counter(p, q)
    jc = n11 / (n11 + n10 + n01)
    return jc


def distanceRC(p, q):
    n11, n10, n01, n00 = n_counter(p, q)
    rc = n11 / (n11 + n10 + n01 + n00)
    return rc


def single_link(ci, cj):
    return min([distanceRC(vi, vj) for vi in ci for vj in cj])


def complete_link(ci, cj):
    return max([distanceSMC(vi, vj) for vi in ci for vj in cj])


def average_link(ci, cj):
    distances = [distanceJC(vi, vj) for vi in ci for vj in cj]
    return sum(distances) / len(distances)


def get_distance_measure(M):
    if M == 0:
        return single_link
    elif M == 1:
        return complete_link
    else:
        return average_link


class AggCluster:
    def __init__(self, data, K, M):
        self.data = data
        self.N = len(data)
        self.K = K
        self.measure = get_distance_measure(M)
        self.clusters = self.init_clusters()

    def init_clusters(self):
        return {data_id: [data_point] for data_id, data_point in enumerate(self.data)}

    def find_closest_clusters(self):
        min_dist = math.inf
        closest_clusters = None
        clusters_ids = list(self.clusters.keys())
        for i, cluster_i in enumerate(clusters_ids[:-1]):
            for j, cluster_j in enumerate(clusters_ids[i + 1:]):
                dist = self.measure(self.clusters[cluster_i], self.clusters[cluster_j])
                if dist < min_dist:
                    min_dist, closest_clusters = dist, (cluster_i, cluster_j)
        return closest_clusters

    def merge_and_form_new_clusters(self, ci_id, cj_id):
        new_clusters = {0: self.clusters[ci_id] + self.clusters[cj_id]}
        for cluster_id in self.clusters.keys():
            if (cluster_id == ci_id) | (cluster_id == cj_id):
                continue
            new_clusters[len(new_clusters.keys())] = self.clusters[cluster_id]
        print(new_clusters)
        return new_clusters

    def run_algorithm(self):
        while len(self.clusters.keys()) > self.K:
            closest_clusters = self.find_closest_clusters()
            self.clusters = self.merge_and_form_new_clusters(*closest_clusters)

    def print(self):
        for id, points in self.clusters.items():
            print("Cluster: {}".format(id))
            for point in points:
                print("    {}".format(point))


dataset = np.array([[1, 1, 1, 0, 1],
                    [1, 0, 1, 0, 0],
                    [1, 1, 0, 1, 1],
                    [1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1],
                    [1, 1, 1, 0, 1]])
print("Single")
agg_hierarchical_clustering = AggCluster(dataset, 1, 0)
agg_hierarchical_clustering.run_algorithm()

print("Complete")
agg_hierarchical_clustering = AggCluster(dataset, 1, 1)
agg_hierarchical_clustering.run_algorithm()

print("Average")
agg_hierarchical_clustering = AggCluster(dataset, 1, 2)
agg_hierarchical_clustering.run_algorithm()
###pictures
ytdist = dataset
Z = hierarchy.linkage(ytdist, 'complete')
plt.figure()
dn = hierarchy.dendrogram(Z)
hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
dn1 = hierarchy.dendrogram(Z, above_threshold_color='y',
                           orientation='top')
dn2 = hierarchy.dendrogram(Z,
                           above_threshold_color='#bcbddc',
                           orientation='right')
hierarchy.set_link_color_palette(None)  # reset to default after use
plt.title("complete")
plt.show()

Z = hierarchy.linkage(ytdist, 'single')
plt.figure()
dn = hierarchy.dendrogram(Z)
hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
dn1 = hierarchy.dendrogram(Z, above_threshold_color='y',
                           orientation='top')
dn2 = hierarchy.dendrogram(Z,
                           above_threshold_color='#bcbddc',
                           orientation='right')
hierarchy.set_link_color_palette(None)  # reset to default after use
plt.title("single")
plt.show()

Z = hierarchy.linkage(ytdist, 'average')
plt.figure()
dn = hierarchy.dendrogram(Z)
hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
dn1 = hierarchy.dendrogram(Z, above_threshold_color='y',
                           orientation='top')
dn2 = hierarchy.dendrogram(Z,
                           above_threshold_color='#bcbddc',
                           orientation='right')
hierarchy.set_link_color_palette(None)  # reset to default after use
plt.title('average')
plt.show()
