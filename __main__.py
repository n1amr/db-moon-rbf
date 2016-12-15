import numpy as np
from dask.array.ufunc import logical_and
from matplotlib import pyplot as plt
from numpy import hstack, vstack, sqrt, newaxis, where, logical_and, array, zeros, ones
from numpy.random import rand, choice


def dbmoon(N=1000, d=1, r=10, w=6, plot=False):
    N //= 2
    N1 = 10 * N
    w2 = w / 2

    tmpl = array([], ndmin=2).reshape((0, 2))
    while True:
        tmp = hstack((2 * (r + w2) * (rand(N1, 1) - 0.5), (r + w2) * rand(N1, 1)))
        dist = sqrt(tmp[:, 0] ** 2 + tmp[:, 1] ** 2)[:, newaxis]
        idx = where(logical_and(dist > (r - w2), dist < (r + w2)))[0]
        tmp = tmp[idx, :]
        tmpl = vstack((tmpl, tmp))
        if tmp.shape[0] >= N:
            break

    data = vstack((
        hstack((tmpl[:N, :], zeros((N, 1)))),
        hstack((tmpl[:N, 0:1] + r, -tmpl[:N, 1:2] - d, ones((N, 1))))
    ))

    if plot:
        plot_data(data)
        plt.axis((- r - w2, 2 * r + w2, -r - w2 - d, r + w2))
        plt.show()

    return data


def plot_data(data):
    idx_class_1 = where(data[:, 2] == 1)[0]
    idx_class_0 = where(data[:, 2] == 0)[0]
    plt.plot(data[idx_class_0, 0], data[idx_class_0, 1], 'r.',
             data[idx_class_1, 0], data[idx_class_1, 1], 'b.')


def k_means_cluster(unlabeled_data, n_clusters):
    n_samples = unlabeled_data.shape[0]

    old_assign = zeros(n_samples) - 2
    assign = zeros(n_samples) - 1

    centroids = unlabeled_data[choice(np.arange(n_samples), n_clusters, replace=False), :]

    max_iters = 10
    for _ in range(max_iters):
        error = 0

        # Find closest cluster
        for i, sample in enumerate(unlabeled_data):
            norms = np.linalg.norm(centroids - sample, axis=1)
            assign[i] = norms.argmin()
            error += norms.min()

        # Compute cluster
        for i, centroid in enumerate(centroids):
            points = unlabeled_data[where(assign == i)[0], :]
            centroids[i] = np.sum(points, axis=0) / points.shape[0]

        if np.all(assign == old_assign):
            break

        old_assign = assign

    return centroids


def main():
    n_samples = 10000
    data = dbmoon(n_samples, d=1, r=10, w=6, plot=False)

    n_dim = 2
    unlabeled_data = data[:, :n_dim]

    n_clusters = 50
    min_centroids = k_means_cluster(unlabeled_data, n_clusters)

    plt.clf()
    plot_data(data)
    plt.plot(min_centroids.T[0], min_centroids.T[1], 'ko')
    plt.show()


if __name__ == '__main__':
    main()
