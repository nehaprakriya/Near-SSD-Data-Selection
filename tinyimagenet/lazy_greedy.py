#!/usr/bin/env python3
print(__doc__)
import matplotlib
# matplotlib.use('TkAgg')

import heapq
import numpy as np
# import pandas as pd
import scipy as sp
import math
from scipy import spatial
import matplotlib.pyplot as plt


class WightedCoverage:

    def __init__(self, S, V, W, alpha=1.):
        self.S = S
        self.V = V
        self.W = W
        self.curVal = 0
        self.gains = []
        self.alpha = alpha
        self.f_norm = self.alpha / self.f_norm(V)
        self.norm = 1. / self.inc(V, [])

    def f_norm(self, sset):
        return np.matmul(self.W[sset], self.S[sset, :].sum(axis=1))

    def inc(self, sset, ndx):
        if not ndx:  # normalization
            return math.log(1 + self.alpha * 1)
        return self.norm * math.log(1 + self.f_norm *
                                    np.matmul(self.W[sset + [ndx]], self.S[sset + [ndx], :].sum(axis=1))
                                    ) - self.curVal

    def add(self, sset, ndx):
        cur_old = self.curVal
        self.curVal = self.norm * math.log(1 + self.f_norm *
                                           np.matmul(self.W[sset + [ndx]], self.S[sset + [ndx], :].sum(axis=1))
                                           )
        self.gains.extend([self.curVal - cur_old])
        return self.curVal


class FacilityLocation1:

    def __init__(self, D, V):
        self.D = D
        self.V = V
        self.curVal = 0
        self.gains = []

    def inc(self, sset, ndx):
        if len(sset + [ndx]) > 1:
            return self.D[:, sset + [ndx]].max(axis=1).sum() - self.curVal
        else:
            return self.D[:, sset + [ndx]].sum() - self.curVal

    def add(self, sset, ndx):
        cur_old = self.curVal
        if len(sset + [ndx]) > 1:
            subset_sim = self.D[:, ndx].sum()
            self.curVal = self.D[:, sset + [ndx]].max(axis=1).sum() - self.gamma * subset_sim
        else:
            self.curVal = self.D[:, sset + [ndx]].sum()
        self.gains.extend([self.curVal - cur_old])
        return self.curVal


class FacilityLocation:

    def __init__(self, D, V, alpha=1., gamma=.0):
        '''
        Args
        - D: np.array, shape [N, N], similarity matrix
        - V: list of int, indices of columns of D
        - alpha: float
        '''
        self.D = D
        self.curVal = 0
        self.curMax = np.zeros(len(D))
        self.gains = []
        self.alpha = alpha
        self.f_norm = self.alpha / self.f_norm(V)
        self.norm = 1. / self.inc(V, [])
        self.gamma = gamma / len(self.D)  # encouraging diversity

    def f_norm(self, sset):
        return self.D[:, sset].max(axis=1).sum()

    def inc(self, sset, ndx):
        if len(sset + [ndx]) > 1:
            if not ndx:  # normalization
                return math.log(1 + self.alpha * 1)
            return self.norm * math.log(1 + self.f_norm * (
                    np.maximum(self.curMax, self.D[:, ndx]).sum() -
                    self.gamma * self.D[sset + [ndx]][:, sset + [ndx]].sum())) - self.curVal
        else:
            return self.norm * math.log(1 + self.f_norm * self.D[:, ndx].sum()) - self.curVal

    def add(self, sset, ndx):
        cur_old = self.curVal
        if len(sset + [ndx]) > 1:
            self.curMax = np.maximum(self.curMax, self.D[:, ndx])
        else:
            self.curMax = self.D[:, ndx]
        self.curVal = self.norm * math.log(1 + self.f_norm * (self.curMax.sum()
                                           - self.gamma * self.D[sset + [ndx]][:, sset + [ndx]].sum()))
        self.gains.extend([self.curVal - cur_old])
        return self.curVal


class FacilityLocation_unnorm:

    def __init__(self, D, V, alpha=1., gamma=.0):
        '''
        Args
        - D: np.array, shape [N, N], similarity matrix
        - V: list of int, indices of columns of D
        - alpha: float
        '''
        self.D = D
        self.curVal = 0
        self.curMax = np.zeros(len(D))
        self.gains = []
        self.alpha = alpha
        # self.f_norm = self.alpha / self.f_norm(V)
        # self.norm = 1. / self.inc(V, [])
        self.gamma = gamma / len(self.D)  # encouraging diversity

    def f_norm(self, sset):
        return self.D[:, sset].max(axis=1).sum()

    def inc(self, sset, ndx):
        if len(sset + [ndx]) > 1:
            # if not ndx:  # normalization
            #     return math.log(1 + self.alpha * 1)
            imp = np.maximum(self.curMax, self.D[:, ndx]).sum() \
                   - (self.gamma * self.D[sset + [ndx]][:, sset + [ndx]].sum()) - self.curVal
            # print(imp)
            if imp < 0:
                2
            return imp
        else:
            imp = self.D[:, ndx].sum() - self.curVal
            # print(imp)
            if imp < 0:
                2
            return imp

    def add(self, sset, ndx):
        cur_old = self.curVal
        if len(sset + [ndx]) > 1:
            self.curMax = np.maximum(self.curMax, self.D[:, ndx])
        else:
            self.curMax = self.D[:, ndx]
        self.curVal = self.curMax.sum() - self.gamma * self.D[sset + [ndx]][:, sset + [ndx]].sum()
        self.gains.extend([self.curVal - cur_old])
        return self.curVal


class NormalizedSoftFacilityLocation:

    def __init__(self, D, V, alpha=1.):
        self.D = D
        self.V = V
        self.curVal = 0
        self.gains = []
        self.alpha = alpha
        self.f_norm = self.alpha / self.f_norm(V)
        self.norm = 1. / self.inc(V, [])

    def f_norm(self, sset):
        sum_exp = np.exp(self.alpha * self.D[:, sset]).sum()
        return 1. / self.alpha * math.log(1 + sum_exp)

    def inc(self, sset, ndx):
        if len(sset + [ndx]) > 1:
            if not ndx:  # normalization
                return math.log(1 + self.alpha * 1)
            sum_exp = np.exp(self.alpha * self.D[:, sset]).sum()
            soft_fl = 1. / self.alpha * math.log(1 + sum_exp)
            soft_norm = self.norm * math.log(1 + self.f_norm * soft_fl)
        else:
            sum_exp = np.exp(self.alpha * self.D[:, sset + [ndx]]).sum()
            soft_fl = 1. / self.alpha * math.log(1 + sum_exp)
            soft_norm = self.norm * math.log(1 + self.f_norm * soft_fl)
        return soft_norm - self.curVal

    def add(self, sset, ndx):
        cur_old = self.curVal
        if not ndx:  # normalization
            sum_exp = np.exp(self.alpha * self.D[:, sset]).sum()
            soft_fl = 1. / self.alpha * math.log(1 + sum_exp)
            soft_norm = self.norm * math.log(1 + self.f_norm * soft_fl)
        else:
            sum_exp = np.exp(self.alpha * self.D[:, sset + [ndx]]).sum()
            soft_fl = 1. / self.alpha * math.log(1 + sum_exp)
            soft_norm = self.norm * math.log(1 + self.f_norm * soft_fl)

        self.curVal = soft_norm
        self.gains.extend([self.curVal - cur_old])
        return self.curVal


class SoftFacilityLocation:

    def __init__(self, D, V, alpha=1.):
        self.D = D
        self.V = V
        self.curVal = 0
        self.gains = []
        self.alpha = alpha
        self.f_norm = self.alpha / self.f_norm(V)
        self.norm = 1. / self.inc(V, [])

    def f_norm(self, sset):
        return self.D[:, sset].max(axis=1).sum()

    def inc(self, sset, ndx):
        if not ndx:
            sum_exp = np.exp(self.alpha * self.D[:, sset]).sum()
        else:
            sum_exp = np.exp(self.alpha * self.D[:, sset + [ndx]]).sum()
        return 1. / self.alpha * math.log(1 + sum_exp) - self.curVal

    def add(self, sset, ndx):
        cur_old = self.curVal
        if not ndx:
            sum_exp = np.exp(self.alpha * self.D[:, sset]).sum()
        else:
            sum_exp = np.exp(self.alpha * self.D[:, sset + [ndx]]).sum()
        self.curVal = 1. / self.alpha * math.log(1 + sum_exp)
        self.gains.extend([self.curVal - cur_old])
        return self.curVal


class FacilityLocation_L:

    def __init__(self, D, V, L, alpha=1.):
        self.D = D
        self.V = V
        self.curVal = 0
        self.gains = []
        self.alpha = alpha
        self.L = L
        self.f_norm = self.alpha / self.f_norm(V)
        self.norm = 1. / self.inc(V, [])

    def f_norm(self, sset):
        return self.D[:, sset].max(axis=1).sum()

    def inc(self, sset, ndx):
        if len(sset + [ndx]) > 1:
            if not ndx:  # normalization
                return math.log(1 + self.alpha * 1)
            return self.norm * math.log(1 + self.f_norm * self.D[:, sset + [ndx]].max(axis=1).sum()) - self.curVal + \
                   self.L[ndx]
        else:
            return self.norm * math.log(1 + self.f_norm * self.D[:, sset + [ndx]].sum()) - self.curVal + self.L[ndx]

    def add(self, sset, ndx):
        cur_old = self.curVal
        if len(sset + [ndx]) > 1:
            self.curVal = self.norm * math.log(1 + self.f_norm * self.D[:, sset + [ndx]].max(axis=1).sum()) + self.L[
                ndx]
        else:
            self.curVal = self.norm * math.log(1 + self.f_norm * self.D[:, sset + [ndx]].sum()) + self.L[ndx]
        self.gains.extend([self.curVal - cur_old])
        return self.curVal


def lazy_greedy(F, ndx, B):
    '''
    Args
    - F: FacilityLocation
    - ndx: indices of all points
    - B: int, number of points to select
    '''
    TOL = 1e-6
    eps = 1e-15
    curVal = 0
    sset = []
    order = []
    vals = []
    for v in ndx:
        marginal = F.inc(sset, v) + eps
        heapq.heappush(order, (1.0 / marginal, v, marginal))

    while order and len(sset) < B:
        el = heapq.heappop(order)
        if not sset:
            improv = el[2]
        else:
            improv = F.inc(sset, el[1]) + eps
            # print(improv)

        # check for uniques elements
        if improv > 0 + eps:
            if not order:
                curVal = F.add(sset, el[1])
                # print curVal
                # print str(len(sset)) + ', ' + str(el[1])
                sset.append(el[1])
                vals.append(curVal)
            else:
                top = heapq.heappop(order)
                if improv >= top[2]:
                    curVal = F.add(sset, el[1])
                    # print curVal
                    # print str(len(sset)) + ', ' + str(el[1])
                    sset.append(el[1])
                    vals.append(curVal)
                else:
                    heapq.heappush(order, (1.0 / improv, el[1], improv))
                heapq.heappush(order, top)
        else:
            2

    # print(sset)
    return sset, vals


def _heappush_max(heap, item):
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap) - 1)


def _heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        heapq._siftup_max(heap, 0)
        return returnitem
    return lastelt


def lazy_greedy_heap(F, V, B):
    curVal = 0
    sset = []
    vals = []

    order = []
    heapq._heapify_max(order)
    [_heappush_max(order, (F.inc(sset, index), index)) for index in V]

    while order and len(sset) < B:
        el = _heappop_max(order)
        improv = F.inc(sset, el[1])

        # check for uniques elements
        if improv >= 0:  # TODO <====
            if not order:
                curVal = F.add(sset, el[1])
                # print curVal
                sset.append(el[1])
                vals.append(curVal)
            else:
                top = _heappop_max(order)
                if improv >= top[0]:
                    curVal = F.add(sset, el[1])
                    # print curVal
                    sset.append(el[1])
                    vals.append(curVal)
                else:
                    _heappush_max(order, (improv, el[1]))
                _heappush_max(order, top)

    # print(str(sset) + ', val: ' + str(curVal))

    return sset, vals


def unconstrained(F, V):
    curVal = 0
    sset = []
    Y = V
    vals = []

    for i in V:
        a = F.inc(sset, i)
        b = F.inc([], list(set(Y) - {i})) - F.inc([], Y)
        if a >= b:
            sset.append(i)
        else:
            Y.remove(i)
    return sset


def test():
    n = 100
    B = 100
    np.random.seed(10)
    X = np.random.rand(n, n)
    D = X * np.transpose(X)

    F = FacilityLocation_unnorm(D, list(range(0, n)), alpha=1, gamma=0)
    sset, vals = lazy_greedy_heap(F, list(np.arange(0, n)), B)
    # print(len(sset), sset)
    # F.sset, F.curVal, F.gains = [], 0, []
    # sset = unconstrained(F, sset)
    # print(len(sset), sset)

    F = FacilityLocation_unnorm(D, list(range(0, n)), alpha=1, gamma=.1)
    sset, vals = lazy_greedy_heap(F, list(np.arange(0, n)), B)
    # print(len(sset), sset)
    # F.sset, F.curVal, F.gains = [], 0, []
    # sset = unconstrained(F, sset)
    # print(len(sset), sset)

    F = FacilityLocation_unnorm(D, list(range(0, n)), alpha=1, gamma=.2)
    sset1, vals = lazy_greedy_heap(F, list(np.arange(0, n)), B)
    # print(len(sset1), sset1)
    # F.sset, F.curVal, F.gains = [], 0, []
    # sset = unconstrained(F, sset)
    # print(len(sset), sset)

    F = FacilityLocation_unnorm(D, list(range(0, n)), alpha=1, gamma=.3)
    sset, vals = lazy_greedy_heap(F, list(np.arange(0, n)), B)
    # print(len(sset), sset)
    # F.sset, F.curVal, F.gains = [], 0, []
    # sset = unconstrained(F, sset)
    # print(len(sset), sset)

    print('--------')
    F = FacilityLocation(D, list(range(0, n)), alpha=1, gamma=.1)
    sset, vals = lazy_greedy_heap(F, list(np.arange(0, n)), B)
    # print(len(sset), sset)
    F = FacilityLocation(D, list(range(0, n)), alpha=1, gamma=.2)
    sset, vals = lazy_greedy_heap(F, list(np.arange(0, n)), B)
    # print(len(sset), sset)
    F = FacilityLocation(D, list(range(0, n)), alpha=1, gamma=.3)
    sset, vals = lazy_greedy_heap(F, list(np.arange(0, n)), B)
    # print(len(sset), sset)

    # print(len(list(set(sset) & set(sset1))))


def test_matrix_mul():
    import datetime, os, pprint, re, sys, time
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import pairwise_distances
    from itertools import permutations
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
    rows = 10
    cols = 10
    num_mats = 8
    num_iterations = 100
    ratios = []
    for iter in range(1, num_iterations):
        max_norm = 0
        min_norm = np.inf
        mats = np.empty([num_mats, rows, rows])
        for ii in range(0, num_mats):
            tmp_mat = np.random.randn(rows, cols)
            mats[ii] = np.dot(tmp_mat, tmp_mat.T)
            mats[ii] = np.eye(rows, rows) - mats[ii]
            mats[ii] = mats[ii] / np.linalg.norm(mats[ii])
        for perm in list(permutations(range(0, num_mats))):
            result = np.eye(rows, rows)
            for ii in perm:
                result = np.dot(result, mats[ii])
            norm = np.linalg.norm(result)
            # print('perm = ',perm,' l2 matrix norm = ',norm)
            if norm > max_norm:
                max_norm = norm
            if norm < max_norm:
                min_norm = norm
        # print('min norm = ',min_norm)
        # print('max norm = ',max_norm)
        # print('max/min ratio = ',max_norm/min_norm)
        print('Finished ', iter, ' of ', num_iterations, ' ratio = ', max_norm / min_norm)
        ratios.append(max_norm / min_norm)
    num_bins = 20
    n, bins, patches = plt.hist(ratios, num_bins, facecolor='blue', alpha=0.5)
    plt.title("Histogram of ratio of max/min norms of products of matrices")
    plt.ylabel("Count")
    plt.xlabel(r"$\frac{\max_\sigma \| \prod_{i \in \sigma} (I-M_i)\|}{\min_\sigma \| \prod_{i \in \sigma} (I-M_i)\|}$")
    plt.show()


def test_matrixmul():
    n = 100
    B = 100
    np.random.seed(10)
    X = np.random.rand(n, n)
    D = X * np.transpose(X)

    F = FacilityLocation_unnorm(D, list(range(0, n)), alpha=1, gamma=0)
    sset, vals = lazy_greedy_heap(F, list(np.arange(0, n)), B)
    # print(len(sset), sset)

    sigmoid = 1. / (1 + np.exp(-X))
    H = np.multiply(X, np.transpose(X))


def cifar(B, num_data):
    G = pd.read_csv('/Users/baharan/Downloads/cifar10/resnet20/1533688447/feats.csv', nrows=num_data).values
    n, dimensions = G.shape
    mymean = np.mean(G, axis=1)
    G = G - np.reshape(mymean, (n, 1)) * np.ones((1, dimensions))
    mynorm = np.linalg.norm(G, axis=1)
    N = np.matmul(np.reshape(mynorm, (n, 1)), np.ones((1, dimensions)))
    G = G / N
    G[np.argwhere(np.isnan(G))] = 0
    G = G + 3 * np.ones((n, dimensions)) / np.sqrt(dimensions)  # shift away from origin
    D = spatial.distance.cdist(G, G, 'euclidean')
    N = np.linalg.norm(G, axis=1)
    N = np.ones((1, n)) * np.reshape(N, (n, 1))
    D = N - D
    F = FacilityLocation(D, range(0, n))
    sset, vals = lazy_greedy(F, xrange(0, n), B)
    # print(sset)
    plt.plot(vals)
    plt.show()
    plt.savefig('cifar10.png')

# cifar(500, 50)
test()
