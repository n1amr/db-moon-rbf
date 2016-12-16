import numpy as np
from matplotlib import pyplot as plt
from numpy import hstack, vstack, sqrt, newaxis, where, array, zeros, ones, exp, tile, logical_and
from numpy.linalg import norm
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

    return data[:, :2], data[:, -1]


def plot_data(X, Y):
    idx_class_0 = where(Y == 0)[0]
    idx_class_1 = where(Y == 1)[0]
    plt.plot(X[idx_class_0, 0], X[idx_class_0, 1], 'b.',
             X[idx_class_1, 0], X[idx_class_1, 1], 'r.')


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


def gaussian_act(X, C, sigma):
    n_clusters = C.shape[0]
    n_samples = X.shape[0]
    X2 = tile(X, (n_clusters, 1, 1))
    C2 = tile(C, (n_samples, 1, 1))
    C2 = C2.transpose([1, 0, 2])
    D = X2 - C2
    N = norm(D, axis=2)
    N = N.T
    return exp(-N ** 2 / (2 * sigma ** 2))


# def gaussian_act2(x. centroids, sigma):
#     n = centroids.shape[0]
#     return exp(-norm(tile(x, (n, 1)) - centroids, axis=1) ** 2 / (2 * sigma ** 2))


def calc_out_o2(W, OH):
    return np.inner(W, OH)



def rbf(X, Y, n_clusters=8):
    n_dim = 2

    # n_clusters = 2

    # X = data[:, :n_dim]
    # Y = data[:, -1]

    n_samples = X.shape[0]

    # Initialization
    C = centroids = k_means_cluster(X, n_clusters)
    W = weights = (rand(1 + n_clusters) - 0.5) / 0.5 * 0.1
    d_max = 0
    for c1 in C:
        for c2 in C:
            d_max = max(d_max, norm(c1 - c2))
    sigma = d_max / sqrt(2 * n_clusters)

    learning_rate_w = 0.2
    learning_rate_c = 0.00001
    learning_rate_sigma = 0.6

    total_error = 0
    for it in range(10000):
        out_h = hstack((ones((n_samples, 1)), gaussian_act(X, C, sigma)))
        out_o = calc_out_o2(W, out_h)

        d_e__d_out_o = - (Y - out_o)

        update_w_sum = np.inner(d_e__d_out_o, out_h.T)

        T1 = ((np.inner(d_e__d_out_o, out_h.T) * W * out_h)[:, 1:])

        C2 = tile(C, (n_samples, 1, 1))
        X2 = tile(X, (n_clusters, 1, 1)).transpose((1, 0, 2))

        T2 = (X2 - C2)
        update_c_sum = (T2 * tile(T1, (2, 1, 1)).transpose(1, 2, 0)).sum(axis=0)

        X2 = tile(X, (n_clusters, 1, 1))
        C2 = tile(C, (n_samples, 1, 1)).transpose([1, 0, 2])
        D = X2 - C2
        N = norm(D, axis=2)
        # N = N.T

        N = hstack((zeros((n_samples, 1)), N.T))

        T1 = N ** 2
        T2 = (np.outer(d_e__d_out_o, W) * T1 / (sigma ** 3) * out_h)
        update_sigma_sum = T2.sum()

        total_error = ((Y - out_o) ** 2 / 2).sum()

        W -= learning_rate_w * update_w_sum / n_samples
        C -= learning_rate_c * update_c_sum / n_samples
        sigma -= learning_rate_sigma * update_sigma_sum / n_samples
        sigma = max(sigma, 0)

        total_error /= n_samples

        if it % 100 == 0:
            print('\niteration #{it}\n=============='.format(**locals()))
            print('weights: {W}'.format(**locals()))
            print('centroids: {C}'.format(**locals()))
            print('sigma: {sigma}'.format(**locals()))
            print('total_error: {total_error}'.format(**locals()))

    return W, C, sigma


def test(X, Y, W, C, sigma):
    outs = []
    misclassified_count = 0

    out_h = hstack((ones((X.shape[0], 1)), gaussian_act(X, C, sigma)))
    out_o = calc_out_o2(W, out_h)


    for i in range(X.shape[0]):
        x = X[i, :]
        y = Y[i]
        out = out_o[i]
        output = 0 if out < 0.4 else 1 if out > 0.6 else None

        if output is None or abs(output - y) > 0.1:
            print('    misclassification: x = {x}, desired = {y}, output = {out}'.format(**locals()))
            misclassified_count += 1
    return misclassified_count


def main():
    np.random.seed(42)

    n_samples = 1000

    X_train, Y_train = dbmoon(n_samples, d=1, r=10, w=6, plot=False)
    X_test, Y_test = dbmoon(n_samples, d=1, r=10, w=6, plot=False)

    n_clusters = 8
    print('using {n_clusters} nodes'.format(**locals()))

    W, C, sigma = rbf(X_train, Y_train, n_clusters)
    e = test(X_test, Y_test, W, C, sigma)

    print('misclassification count = {e}'.format(**locals()))


if __name__ == '__main__':
    main()
