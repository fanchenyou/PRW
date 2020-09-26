# -*- coding: utf-8 -*-

#############################################################
# Experiment for testing algorithm speed of four different lr
#############################################################

import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.ticker as ticker

from PRW import ProjectionRobustWasserstein
from Optimization.riemann_adap import RiemmanAdaptive

import pickle


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


ds = [25, 50, 100, 250, 500]  # , 1000]  # Dimensions for which to compute the SRW computation time
nb_ds = len(ds)
n = 100  # Number of points in the measures
k = 2  # Dimension parameter
reg = 0.2  # Entropic regularization strength
max_iter = 500  # Maximum number of iterations
max_iter_sinkhorn = 30  # Maximum number of iterations in Sinkhorn
threshold = 1e-3  # Stopping threshold
threshold_sinkhorn = 1e-3  # Stopping threshold in Sinkhorn
nb_exp = 50  # Number of experiments

lrs = [0.005, 0.01, 0.05, 0.1]

times_PRW = np.zeros((2, len(lrs), nb_exp, nb_ds))

np.random.seed(357)

tic = time.time()
tac = time.time()

if 1 == 1:
    for t in range(nb_exp):
        print('iter', t)
        for ind_lr in range(len(lrs)):
            for ind_d in range(nb_ds):
                d = ds[ind_d]
                lr = lrs[ind_lr]
                # print(d)

                a, b, X, Y = fragmented_hypercube(n, d, dim=2)

                reg = 0.2
                if d >= 250:
                    reg = 0.5
                if lr > 0.01:
                    reg *= 10


                #print('PRW(1)', lr)
                algo = RiemmanAdaptive(reg=reg, step_size_0=None, max_iter=max_iter,
                                       max_iter_sinkhorn=max_iter_sinkhorn,
                                       threshold=threshold, threshold_sinkhorn=threshold_sinkhorn, use_gpu=False)
                PRW = ProjectionRobustWasserstein(X, Y, a, b, algo, k)
                tic = time.time()
                PRW.run(0, lr=lr, beta=None)
                tac = time.time()
                times_PRW[0, ind_lr, t, ind_d] = tac - tic

                #print('PRW(2)',lr)
                algo = RiemmanAdaptive(reg=reg, step_size_0=None, max_iter=max_iter,
                                       max_iter_sinkhorn=max_iter_sinkhorn,
                                       threshold=threshold, threshold_sinkhorn=threshold_sinkhorn, use_gpu=False)
                PRW = ProjectionRobustWasserstein(X, Y, a, b, algo, k)
                tic = time.time()
                PRW.run(1, lr=lr, beta=0.8)
                tac = time.time()
                times_PRW[1, ind_lr, t, ind_d] = tac - tic

    with open('./results/exp4_robust_lr_time_2.pkl', 'wb') as f:
        pickle.dump(times_PRW, f)

else:
    with open('./results/exp4_robust_lr_time_2.pkl', 'rb') as f:
        times_PRW = pickle.load(f)

line_styles = ['-', '-',':', ':']
captions = ['PRW (RGAS)  ', 'PRW (RAGAS)']

for t in range(2):
    plt.figure(figsize=(17, 9))
    for ind_lr in range(len(lrs)):

        cap = '%s lr=%.3f' % (captions[t], lrs[ind_lr])

        time_t_lr = times_PRW[t, ind_lr, :, :]
        times_mean = np.mean(time_t_lr, axis=0)
        times_min = np.min(time_t_lr, axis=0)
        times_max = np.max(time_t_lr, axis=0)

        if ind_lr in [1, 3]:
            mean, = plt.loglog(ds, times_mean, 'C%d' % (t + 1,), ls=line_styles[ind_lr], lw=6, ms=8, label=cap)
        elif t == 0:
            mean, = plt.loglog(ds, times_mean, ls=line_styles[ind_lr], c='m', lw=6, ms=8, label=cap)
        elif t == 1:
            mean, = plt.loglog(ds, times_mean, ls=line_styles[ind_lr], c='b', lw=6, ms=8, label=cap)

        col = mean.get_color()
        plt.fill_between(ds, times_min, times_max, facecolor=col, alpha=0.15)

    plt.xlabel('Dimension', fontsize=25)
    plt.ylabel('Execution time in seconds', fontsize=25)
    plt.legend(loc='best', fontsize=25, handlelength=3)
    plt.xticks(ds, fontsize=20)
    plt.ylim((1e-3, 10))
    # plt.yticks(fontsize=20)
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
    plt.grid(ls=':')
    plt.savefig('figs/exp4_computation_time_lr_%d.png' % (t,))
    plt.show()
    plt.close()
