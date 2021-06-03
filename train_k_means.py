import numpy as np
from copy import deepcopy

rng = np.random.default_rng()


def k_means(data, k):
    print(data.shape)
    ct = 0
    clusters = np.array([rng.random(7) for _ in range(k)])
    while True:
        ct += 1
        prev_clusters = deepcopy(clusters)
        norms = np.concatenate([np.linalg.norm(data - cl, axis=1).reshape((-1, 1)) for cl in clusters], axis=1)
        mins = np.argmin(norms, axis=1)

        clusters = np.array([(data.T @ (mins == i)) / np.sum(mins == i) for i in range(3)])

        if all(sum(i - j) == 0 for i, j in zip(clusters, prev_clusters)):
            break

        if ct > 100:
            print('Convergence Failed')
            break

        if clusters[0][0] > clusters[1][0]:
            clusters = clusters[-1::-1]
    return clusters

def k_means_ss(data, labeled_data, k, alpha):
    ct = 0
    clusters = np.array([rng.random(7) for _ in range(k)])
    while True:
        ct += 1
        prev_clusters = deepcopy(clusters)
        norms = np.concatenate([np.linalg.norm(data - cl, axis=1).reshape((-1, 1)) for cl in clusters], axis=1)
        mins = np.argmin(norms, axis=1)

        clusters = np.array([((data.T @ (mins == i)) + alpha * np.sum(labeled_data[labeled_data[:,-1] == i, :-1], axis=0))/
                             (np.sum(mins == i) + alpha * np.sum(labeled_data[:,-1] == i))
                             for i in range(3)])

        if all(sum(i - j) == 0 for i, j in zip(clusters, prev_clusters)):
            break

        if ct > 100:
            print('Convergence Failed')
            break

        if clusters[0][0] > clusters[1][0]:
            clusters = clusters[-1::-1]
    return clusters


train_data = np.loadtxt('./train_data.txt')

'''clusters = k_means(train_data[:, :-1], 3)
print('Clusters: ')
print(clusters)
np.savetxt('./clusters.txt', clusters)'''

a = 200
clusters_ss = k_means_ss(train_data[:-a, :-1], train_data[-a:], 3, 5)

print('Clusters Semi-Sup: ')
print(clusters_ss)
exit(0)
np.savetxt('./clusters_ss.txt', clusters_ss)
