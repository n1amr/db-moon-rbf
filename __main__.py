from dask.array.ufunc import logical_and
from matplotlib import pyplot as plt
from numpy import hstack, vstack, sqrt, newaxis, where, logical_and, array, zeros, ones
from numpy.random import rand


def dbmoon(N=1000, d=1, r=10, w=6, plot=False):
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
        plt.plot(data[:N, 0], data[:N, 1], 'r.',
                 data[N:, 0], data[N:, 1], 'b.')
        plt.axis((- r - w2, 2 * r + w2, -r - w2 - d, r + w2))
        plt.show()

    return data


def main():
    data = dbmoon(1000, 1, 10, 6, plot=True)


if __name__ == '__main__':
    main()
