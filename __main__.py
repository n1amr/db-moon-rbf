import numpy as np
from dask.array.ufunc import logical_and
from matplotlib import pyplot as plt
from numpy import hstack, vstack, sqrt, newaxis, where, logical_and, array, zeros, ones, exp, tile
from numpy.random import rand, choice
from numpy.linalg import norm


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
    plt.plot(data[idx_class_0, 0], data[idx_class_0, 1], 'b.',
             data[idx_class_1, 0], data[idx_class_1, 1], 'r.')


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


def gaussian_act(x, centroids, sigma):
    n = centroids.shape[0]
    return exp(-norm(tile(x, (n, 1)) - centroids, axis=1) ** 2 / (2 * sigma ** 2))


def calc_out_o(weights, out_h):
    # out_h = hstack((1, out_h))
    return np.sum(weights * out_h)


def rbf(data, n_clusters=8):
    n_dim = 2

    # n_clusters = 2

    X = data[:, :n_dim]
    Y = data[:, -1]

    n_samples = X.shape[0]

    # Initialization
    C = centroids = k_means_cluster(X, n_clusters)
    W = weights = (rand(1 + n_clusters) - 0.5) / 0.5 * 0.1
    d_max = 0
    for c1 in C:
        for c2 in C:
            d_max = max(d_max, norm(c1 - c2))
    sigma = d_max / sqrt(2 * n_clusters)

    learning_rate_w = 1
    learning_rate_c = 1
    learning_rate_sigma = 2

    total_error = 0
    # Forward pass
    for it in range(1000):
        if it % 10 == 0:
            print('\niteration #{it}\n=============='.format(**locals()))
            print('weights: {W}'.format(**locals()))
            print('centroids: {C}'.format(**locals()))
            print('sigma: {sigma}'.format(**locals()))
            print('total_error: {total_error}'.format(**locals()))

        update_w_sum = 0
        update_c_sum = 0
        update_sigma_sum = 0
        total_error = 0
        for i in range(n_samples):
            x = X[i, :]
            y = Y[i]
            out_h = gaussian_act(x, C, sigma)
            out_h = hstack((1, out_h))  # H x 1
            out_o = calc_out_o(W, out_h)  # 1 x 1
            e = (y - out_o) ** 2 / 2

            d_e__d_w = - (y - out_o) * out_h  # O x 1

            t1 = (- (y - out_o) * out_h * W * out_h)[1:]
            t2 = tile(x, (n_clusters, 1)) - C
            d_e__d_c = (t2.T * t1).T

            n = n_clusters
            hstack((0, norm(tile(x, (n_clusters, 1)) - centroids, axis=1) ** 2))

            t1 = hstack((0, norm(tile(x, (n, 1)) - centroids, axis=1) ** 2))
            t2 = - (y - out_o) * W * t1 / (sigma ** 3) * out_h
            d_e__d_sigma = np.sum(t2)

            update_w_sum += d_e__d_w
            update_c_sum += d_e__d_c
            update_sigma_sum += d_e__d_sigma
            total_error += e

            # Back propagation

        W -= learning_rate_w * update_w_sum / n_samples
        C -= learning_rate_c * update_c_sum / n_samples
        sigma -= learning_rate_sigma * update_sigma_sum / n_samples
        sigma = max(sigma, 0)

        total_error /= n_samples

    plt.clf()
    plot_data(data)
    plt.plot(centroids.T[0], centroids.T[1], 'ko')
    plt.show()

    return W, C, sigma


def main():
    n_samples = 1000
    data = dbmoon(n_samples, d=1, r=10, w=6, plot=False)
    # data = array([
    #     [0, 0.01, 0],
    #     [0.01, 0, 0],
    #     [0, -0.01, 0],
    #     [-0.01, 0, 0],
    #     [10, 10.01, 1],
    #     [10.01, 10, 1],
    #     [10, 9.99, 1],
    #     [9.99, 10, 1],
    # ])
    W, C, sigma = rbf(data, n_clusters=10)

    data = dbmoon(n_samples, d=1, r=10, w=6, plot=False)
    X = data[:, :2]
    Y = data[:, -1]

    outs = []
    for i in range(X.shape[0]):
        x = X[i, :]
        y = Y[i]
        out_h = gaussian_act(x, C, sigma)
        out_h = hstack((1, out_h))  # H x 1
        out_o = calc_out_o(W, out_h)  # 1 x 1
        outs.append(out_o)
        out_o = 0 if out_o < 0.3 else 1 if out_o > 0.7 else 100000
        if abs(out_o - y) > 0.1:
            print('x = {x}, y = {y}, out = {out_o}'.format(**locals()))


if __name__ == '__main__':
    main()
