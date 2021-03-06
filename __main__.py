__author__ = 'Amr Alaa'

import numpy as np
from matplotlib import pyplot as plt
from numpy import hstack, vstack, sqrt, newaxis, where, array, zeros, ones, exp, tile, logical_and
from numpy.linalg import norm
from numpy.random import rand, choice


def dbmoon(N=1000, d=1, r=10, w=6):
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


def gaussian_act(x, centroids, sigma):
    n = centroids.shape[0]
    return exp(-norm(tile(x, (n, 1)) - centroids, axis=1) ** 2 / (2 * sigma ** 2))


def calc_out_o(weights, out_h):
    return np.sum(weights * out_h)


def rbf(X, Y, n_clusters=8):
    n_dim = 2

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
    learning_rate_sigma = 1

    total_error = 0
    for it in range(80):
        update_w_sum = 0
        update_c_sum = 0
        update_sigma_sum = 0
        total_error = 0
        for i in range(n_samples):
            x = X[i, :]
            y = Y[i]

            # Forward pass
            out_h = hstack((1, gaussian_act(x, C, sigma)))
            out_o = calc_out_o(W, out_h)

            # Back propagation
            e = (y - out_o) ** 2 / 2

            d_e__d_out_o = - (y - out_o)
            d_e__d_w = d_e__d_out_o * out_h

            t1 = (d_e__d_out_o * out_h * W * out_h)[1:]
            t2 = tile(x, (n_clusters, 1)) - C
            d_e__d_c = (t2.T * t1).T

            t1 = hstack((0, norm(tile(x, (n_clusters, 1)) - centroids, axis=1) ** 2))
            t2 = d_e__d_out_o * W * t1 / (sigma ** 3) * out_h
            d_e__d_sigma = np.sum(t2)

            update_w_sum += d_e__d_w
            update_c_sum += d_e__d_c
            update_sigma_sum += d_e__d_sigma
            total_error += e

        W -= learning_rate_w * update_w_sum / n_samples
        C -= learning_rate_c * update_c_sum / n_samples
        sigma -= learning_rate_sigma * update_sigma_sum / n_samples
        sigma = max(sigma, 0)

        total_error /= n_samples

        if (it + 1) % 20 == 0:
            print('\niteration #{it}\n=============='.format(it=it + 1))
            print('weights = {W}'.format(**locals()))
            print('centroids = \n{C}'.format(**locals()))
            print('sigma = {sigma}'.format(**locals()))
            print('total error = {total_error}'.format(**locals()))

    return W, C, sigma


def test(X, Y, W, C, sigma):
    outs = []
    misclassified_count = 0
    error = 0
    m = X.shape[0]
    for i in range(m):
        x = X[i, :]
        y = Y[i]
        out_h = gaussian_act(x, C, sigma)
        out_h = hstack((1, out_h))
        out_o = calc_out_o(W, out_h)
        error += (out_o - y) ** 2
        outs.append(out_o)
        output = 0 if out_o < 0.4 else 1 if out_o > 0.6 else None

        if output is None or abs(output - y) > 0.1:
            print('    misclassification: x = {x}, desired = {y}, output = {out_o}'.format(**locals()))
            misclassified_count += 1
    return misclassified_count, error / (2 * m)


def main():
    np.random.seed(42)

    n_samples = 1000

    X_train, Y_train = dbmoon(n_samples, d=1, r=10, w=6)
    X_test, Y_test = dbmoon(n_samples, d=1, r=10, w=6)

    opt_min = 2
    opt_max = n_samples // 100
    ERROR_THRESHOLD = 0.02
    MISSRATE_THRESHOLD = 0.02
    while opt_max - opt_min > 0:
        n_clusters = (opt_min + opt_max) // 2
        print('using {n_clusters} nodes'.format(**locals()))

        W, C, sigma = rbf(X_train, Y_train, n_clusters)
        miss_count, error = test(X_test, Y_test, W, C, sigma)

        print('misclassification count = {miss_count}'.format(**locals()))
        if error <= ERROR_THRESHOLD and miss_count / n_samples < MISSRATE_THRESHOLD:
            opt_max = n_clusters
            opt_W, opt_C, opt_sigma = (W, C, sigma)
            opt_miss_count, opt_error = (miss_count, error)
        else:
            opt_min = n_clusters + 1

    opt = opt_max

    print('\n\nsummary:\n' + '=' * 10)
    print('optimal weights = {opt_W}'.format(**locals()))
    print('optimal centroids = \n{opt_C}'.format(**locals()))
    print('optimal sigma = {opt_sigma}'.format(**locals()))
    print('\n' + '=' * 10)
    print('optimal number of hidden layer nodes is {opt}'.format(**locals()))
    print('classification error = {error:2.2f}%'.format(error=opt_error * 100))
    print('misclassification ratio on test set = {opt_miss_count}/{n_samples}'.format(**locals()))


if __name__ == '__main__':
    main()
