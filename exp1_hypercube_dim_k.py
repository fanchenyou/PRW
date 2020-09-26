#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################
# Experiment estimate PRW mean error with different dimensions
################################################################
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle

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

d = 100  # Total dimension
#k = 2  # k* = 2 and compute SRW with k = 2
nb_exp = 30
n = 100
kstars = [2, 4, 7, 10]
colors = ['b', 'orange', 'g', 'r']
maxK = 30
values = np.zeros((2, len(kstars), maxK, nb_exp))

proj = np.zeros((d, d))  # Real optimal subspace
proj[0, 0] = 1
proj[1, 1] = 1

np.random.seed(123)

if 1==1:
    for t in range(nb_exp):
        print(t)
        for i, kstar in enumerate(kstars):
            for kdim in range(1, maxK + 1):
                a, b, X, Y = fragmented_hypercube(n, d, dim=kstar)

                algo = RiemmanAdaptive(reg=0.2, step_size_0=None, max_iter=30, threshold=1e-4,
                                       max_iter_sinkhorn=30,
                                       threshold_sinkhorn=1e-04, use_gpu=False)
                PRW = ProjectionRobustWasserstein(X, Y, a, b, algo, kdim)
                PRW.run(0, lr=0.01, beta=None)
                values[0, i, kdim - 1, t] = np.abs(PRW.get_value())

                algo = RiemmanAdaptive(reg=0.2, step_size_0=None, max_iter=30, threshold=1e-4,
                                       max_iter_sinkhorn=30,
                                       threshold_sinkhorn=1e-04, use_gpu=False)
                PRW = ProjectionRobustWasserstein(X, Y, a, b, algo, kdim)
                PRW.run(1, lr=0.005, beta=0.8)
                values[1, i, kdim - 1, t] = np.abs(PRW.get_value())

    with open('./results/exp1_hypercube_dim_k.pkl', 'wb') as f:
        pickle.dump(values, f)

else:
    with open('./results/exp1_hypercube_dim_k.pkl', 'rb') as f:
        values = pickle.load(f)

plt.figure(figsize=(20, 8))

Xs = list(range(1, maxK + 1))
line_styles = ['-', '--']
captions = ['RGAS', 'RAGAS']
for t in range(2):
    for i, kstar in enumerate(kstars):
        values_mean = np.mean(values[t, i, :, :], axis=1)
        values_min = np.min(values[t, i, :, :], axis=1)
        values_max = np.max(values[t, i, :, :], axis=1)

        mean, = plt.plot(Xs, values_mean, ls=line_styles[t],
                         c=colors[i], lw=4, ms=20,
                         label='$k^*=%d$, %s' % (kstar,captions[t]))
        col = mean.get_color()
        plt.fill_between(Xs, values_min, values_max, facecolor=col, alpha=0.15)

for i in range(len(kstars)):
    ks = kstars[i]
    vm1 = np.mean(values[0, i, ks, :], axis=0)
    vm2 = np.mean(values[1, i, ks, :], axis=0)
    print(vm1,vm2)
    tt = max(vm1,vm2)
    plt.plot([ks, ks], [0, tt], color=colors[i], linestyle='--')


plt.xlabel('Dimension k', fontsize=25)
plt.ylabel('PRW values', fontsize=25)
plt.ylabel('$P_k^2(\hat\mu, \hat\\nu)$', fontsize=25)
plt.xticks(Xs, fontsize=20)
plt.yticks(np.arange(10, 70+1, 10), fontsize=20)
plt.legend(loc='best', fontsize=18, ncol=2)
plt.ylim(0)
plt.title('$P_k^2(\hat\mu, \hat\\nu)$ depending on dimension k', fontsize=30)
plt.minorticks_on()
plt.grid(ls=':')
plt.savefig('figs/exp1_dim_k.png')
plt.show()
