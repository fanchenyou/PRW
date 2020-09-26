# -*- coding: utf-8 -*-

#################################################
# Experiment for testing algorithm speed on CPU
#################################################

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

from Optimization.sinkhorn import sinkhorn_knopp

from SRW import SubspaceRobustWasserstein
from Optimization.frankwolfe import FrankWolfe

from PRW import ProjectionRobustWasserstein
from Optimization.riemann_adap import RiemmanAdaptive


def T(x, d, dim=2):
    assert dim <= d
    assert dim >= 1
    assert dim == int(dim)
    return x + 2 * np.sign(x) * np.array(dim * [1] + (d - dim) * [0])


def fragmented_hypercube(n, d, dim):
    assert dim <= d
    assert dim >= 1
    assert dim == int(dim)

    a = (1. / n) * np.ones(n)
    b = (1. / n) * np.ones(n)

    # First measure : uniform on the hypercube
    X = np.random.uniform(-1, 1, size=(n, d))

    # Second measure : fragmentation
    Y = T(np.random.uniform(-1, 1, size=(n, d)), d, dim)

    return a, b, X, Y


ds = [10, 25, 50, 100, 250, 500]  # , 1000]  # Dimensions for which to compute the SRW computation time
nb_ds = len(ds)
n = 100  # Number of points in the measures
k = 2  # Dimension parameter
reg = 0.2  # Entropic regularization strength
max_iter = 500  # Maximum number of iterations
max_iter_sinkhorn = 30  # Maximum number of iterations in Sinkhorn
threshold = 1e-3  # Stopping threshold
threshold_sinkhorn = 1e-3  # Stopping threshold in Sinkhorn
nb_exp = 50  # Number of experiments

times_SRW = np.zeros((nb_exp, nb_ds))
times_W = np.zeros((nb_exp, nb_ds))
times_PRW_1 = np.zeros((nb_exp, nb_ds))
times_PRW_2 = np.zeros((nb_exp, nb_ds))

np.random.seed(321)

if 1 == 2:
    tic = time.time()
    tac = time.time()
    for t in range(nb_exp):
        print('iter', t)

        for ind_d in range(nb_ds):
            d = ds[ind_d]

            a, b, X, Y = fragmented_hypercube(n, d, dim=2)

            reg = 0.2
            if d >= 250:
                reg = 0.5

            # print('W')
            tic = time.time()
            ones = np.ones((n, n))
            C = np.diag(np.diag(X.dot(X.T))).dot(ones) + ones.dot(np.diag(np.diag(Y.dot(Y.T)))) - 2 * X.dot(Y.T)
            OT_plan = sinkhorn_knopp(a, b, C, reg, numItermax=max_iter_sinkhorn, stopThr=threshold_sinkhorn)
            tac = time.time()
            times_W[t, ind_d] = tac - tic

            # print('SRW')
            algo = FrankWolfe(reg=reg, step_size_0=None, max_iter=max_iter, max_iter_sinkhorn=max_iter_sinkhorn,
                              threshold=threshold, threshold_sinkhorn=threshold_sinkhorn, use_gpu=False)
            SRW = SubspaceRobustWasserstein(X, Y, a, b, algo, k)
            tic = time.time()
            SRW.run()
            tac = time.time()
            times_SRW[t, ind_d] = tac - tic

            # print('PRW(1)')
            algo = RiemmanAdaptive(reg=reg, step_size_0=None, max_iter=max_iter, max_iter_sinkhorn=max_iter_sinkhorn,
                                   threshold=threshold, threshold_sinkhorn=threshold_sinkhorn, use_gpu=False)
            PRW = ProjectionRobustWasserstein(X, Y, a, b, algo, k)
            tic = time.time()
            PRW.run(0, lr=0.01, beta=None)
            tac = time.time()
            times_PRW_1[t, ind_d] = tac - tic

            # print('PRW(2)')
            algo = RiemmanAdaptive(reg=reg, step_size_0=None, max_iter=max_iter, max_iter_sinkhorn=max_iter_sinkhorn,
                                   threshold=threshold, threshold_sinkhorn=threshold_sinkhorn, use_gpu=False)
            PRW = ProjectionRobustWasserstein(X, Y, a, b, algo, k)
            tic = time.time()
            PRW.run(1, lr=0.01, beta=0.8)
            tac = time.time()
            times_PRW_2[t, ind_d] = tac - tic

    print(times_SRW.shape, times_W.shape)

    with open('./results/exp4_computation_time.pkl', 'wb') as f:
        pickle.dump([times_SRW, times_W, times_PRW_1, times_PRW_2], f)

else:

    with open('./results/exp4_computation_time.pkl', 'rb') as f:
        times_SRW, times_W, times_PRW_1, times_PRW_2 = pickle.load(f)

times_SRW_mean = np.mean(times_SRW, axis=0)
times_SRW_min = np.min(times_SRW, axis=0)
times_SRW_10 = np.percentile(times_SRW, 10, axis=0)
times_SRW_25 = np.percentile(times_SRW, 25, axis=0)
times_SRW_75 = np.percentile(times_SRW, 75, axis=0)
times_SRW_90 = np.percentile(times_SRW, 90, axis=0)
times_SRW_max = np.max(times_SRW, axis=0)

# times_W_mean = np.mean(times_W, axis=0)
# times_W_min = np.min(times_W, axis=0)
# times_W_10 = np.percentile(times_W, 10, axis=0)
# times_W_25 = np.percentile(times_W, 25, axis=0)
# times_W_75 = np.percentile(times_W, 75, axis=0)
# times_W_90 = np.percentile(times_W, 90, axis=0)
# times_W_max = np.max(times_W, axis=0)

times_PRW_1_mean = np.mean(times_PRW_1, axis=0)
times_PRW_1_min = np.min(times_PRW_1, axis=0)
times_PRW_1_10 = np.percentile(times_PRW_1, 10, axis=0)
times_PRW_1_25 = np.percentile(times_PRW_1, 25, axis=0)
times_PRW_1_75 = np.percentile(times_PRW_1, 75, axis=0)
times_PRW_1_90 = np.percentile(times_PRW_1, 90, axis=0)
times_PRW_1_max = np.max(times_PRW_1, axis=0)

times_PRW_2_mean = np.mean(times_PRW_2, axis=0)
times_PRW_2_min = np.min(times_PRW_2, axis=0)
times_PRW_2_10 = np.percentile(times_PRW_2, 10, axis=0)
times_PRW_2_25 = np.percentile(times_PRW_2, 25, axis=0)
times_PRW_2_75 = np.percentile(times_PRW_2, 75, axis=0)
times_PRW_2_90 = np.percentile(times_PRW_2, 90, axis=0)
times_PRW_2_max = np.max(times_PRW_2, axis=0)

import matplotlib.ticker as ticker

plt.figure(figsize=(16, 8))

# mean, = plt.loglog(ds[1:], times_W_mean[1:], 'o-', lw=8, ms=10, label='Wasserstein')
# col = mean.get_color()
# plt.fill_between(ds[1:], times_W_25[1:], times_W_75[1:], facecolor=col, alpha=0.3)
# plt.fill_between(ds[1:], times_W_10[1:], times_W_90[1:], facecolor=col, alpha=0.2)
# plt.fill_between(ds[1:], times_W_min[1:], times_W_max[1:], facecolor=col, alpha=0.15)

mean, = plt.loglog(ds[1:], times_SRW_mean[1:], 'o-', lw=3, ms=10, label='SRW (FW)')
col = mean.get_color()
plt.fill_between(ds[1:], times_SRW_25[1:], times_SRW_75[1:], facecolor=col, alpha=0.3)
plt.fill_between(ds[1:], times_SRW_10[1:], times_SRW_90[1:], facecolor=col, alpha=0.2)
plt.fill_between(ds[1:], times_SRW_min[1:], times_SRW_max[1:], facecolor=col, alpha=0.15)

mean, = plt.loglog(ds[1:], times_PRW_1_mean[1:], 'o-', lw=3, ms=10, label='PRW (RGAS)')
col = mean.get_color()
plt.fill_between(ds[1:], times_PRW_1_25[1:], times_PRW_1_75[1:], facecolor=col, alpha=0.3)
plt.fill_between(ds[1:], times_PRW_1_10[1:], times_PRW_1_90[1:], facecolor=col, alpha=0.2)
plt.fill_between(ds[1:], times_PRW_1_min[1:], times_PRW_1_max[1:], facecolor=col, alpha=0.15)

mean, = plt.loglog(ds[1:], times_PRW_2_mean[1:], 'o-', lw=3, ms=10, label='PRW (RAGAS)')
col = mean.get_color()
plt.fill_between(ds[1:], times_PRW_2_25[1:], times_PRW_2_75[1:], facecolor=col, alpha=0.3)
plt.fill_between(ds[1:], times_PRW_2_10[1:], times_PRW_2_90[1:], facecolor=col, alpha=0.2)
plt.fill_between(ds[1:], times_PRW_2_min[1:], times_PRW_2_max[1:], facecolor=col, alpha=0.15)

plt.xlabel('Dimension', fontsize=25)
plt.ylabel('Execution time in seconds', fontsize=25)
plt.xticks(ds[1:], fontsize=20)
plt.yticks(fontsize=20)
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
plt.grid(ls=':')
plt.legend(loc='best', fontsize=18, handlelength=3)
plt.savefig('figs/exp4_computation_time.png')
plt.show()
