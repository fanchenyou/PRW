#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###########################################################
# Experiment to estimate PRW mean error on hypercube data
###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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


d = 30  # Total dimension
k = 2  # k* = 2 and compute SRW with k = 2
nb_exp = 100  # Do 100 experiments
ns = [25, 50, 100, 250, 500, 1000]  # Compute SRW between measures with 'n' points for 'n' in 'ns'

values = np.zeros((2, len(ns), nb_exp))
values_subspace = np.zeros((2, len(ns), nb_exp))

proj = np.zeros((d, d))  # Real optimal subspace
proj[0, 0] = 1
proj[1, 1] = 1

np.random.seed(357)


if 1==2:
    for indn in range(len(ns)):
        n = ns[indn]
        # Sample nb_exp times
        for t in range(nb_exp):
            a, b, X, Y = fragmented_hypercube(n, d, dim=2)

            # Riemann Gradient
            algo = RiemmanAdaptive(reg=0.2, step_size_0=None, max_iter=30, threshold=1e-3,
                                   max_iter_sinkhorn=30,
                                   threshold_sinkhorn=1e-03, use_gpu=False)
            PRW = ProjectionRobustWasserstein(X, Y, a, b, algo, k)

            Omega, pi, maxmin_values = PRW.run(0, lr=0.01, beta=None)
            values[0, indn, t] = np.abs(8 - PRW.get_value())
            values_subspace[0, indn, t] = np.linalg.norm(Omega - proj)


            # Riemann Adaptive Gradient
            algo = RiemmanAdaptive(reg=0.2, step_size_0=None, max_iter=30, threshold=1e-3,
                                   max_iter_sinkhorn=30,
                                   threshold_sinkhorn=1e-03, use_gpu=False)
            PRW = ProjectionRobustWasserstein(X, Y, a, b, algo, k)

            Omega, pi, maxmin_values = PRW.run(1, lr=0.01, beta=0.8)
            values[1, indn, t] = np.abs(8 - PRW.get_value())
            values_subspace[1, indn, t] = np.linalg.norm(Omega - proj)



        print('(RG)  n =', n, '/', np.mean(values[0, indn, :]), np.mean(values_subspace[0, indn, :]))
        print('(RAG) n =', n, '/', np.mean(values[1, indn, :]), np.mean(values_subspace[1, indn, :]))

    with open('./results/exp1_hypercube_value.pkl', 'wb') as f:
        pickle.dump([values, values_subspace], f)

else:
    with open('./results/exp1_hypercube_value.pkl', 'rb') as f:
        values, values_subspace = pickle.load(f)

captions = ['PRW (RGAS)', 'PRW (RAGAS)']

plt.figure(figsize=(12, 8))
for t in range(2):
    values_mean = np.mean(values[t,:,:], axis=1)
    values_min = np.min(values[t,:,:], axis=1)
    values_10 = np.percentile(values[t,:,:], 10, axis=1)
    values_25 = np.percentile(values[t,:,:], 25, axis=1)
    values_75 = np.percentile(values[t,:,:], 75, axis=1)
    values_90 = np.percentile(values[t,:,:], 90, axis=1)
    values_max = np.max(values[t,:,:], axis=1)

    mean, = plt.semilogy(ns, values_mean, 'o-', lw=4, ms=11,
                       label=captions[t])
    col = mean.get_color()
    plt.fill_between(ns, values_25, values_75, facecolor=col, alpha=0.3)
    plt.fill_between(ns, values_10, values_90, facecolor=col, alpha=0.2)

plt.xlabel('Number of points', fontsize=25)
plt.ylabel('$|W^2(\mu,\\nu) - P_2^2(\hat\mu, \hat\\nu)|$', fontsize=25)
plt.legend(loc='best', fontsize=25)
plt.title('Mean estimation error', fontsize=30)

plt.xticks(ns, fontsize=20)
plt.yticks(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]), fontsize=20)
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
plt.grid(ls=':')
plt.savefig('figs/exp1_hypercube_value_1.png')
plt.show()
plt.close()
plt.clf()

plt.figure(figsize=(12, 8))
for t in range(2):
    values_subspace_mean = np.mean(values_subspace[t,:,:], axis=1)
    values_subspace_min = np.min(values_subspace[t,:,:], axis=1)
    values_subspace_10 = np.percentile(values_subspace[t,:,:], 10, axis=1)
    values_subspace_25 = np.percentile(values_subspace[t,:,:], 25, axis=1)
    values_subspace_75 = np.percentile(values_subspace[t,:,:], 75, axis=1)
    values_subspace_90 = np.percentile(values_subspace[t,:,:], 90, axis=1)
    values_subspace_max = np.max(values_subspace[t,:,:], axis=1)

    mean, = plt.loglog(ns, values_subspace_mean, 'o-', lw=4, ms=11,
                       label=captions[t])
    col = mean.get_color()
    plt.fill_between(ns, values_subspace_25, values_subspace_75, facecolor=col, alpha=0.3)
    plt.fill_between(ns, values_subspace_10, values_subspace_90, facecolor=col, alpha=0.2)
    plt.fill_between(ns, values_subspace_min, values_subspace_max, facecolor=col, alpha=0.15)

plt.xlabel('Number of points', fontsize=25)
plt.ylabel('$||\Omega^* - \widehat\Omega||_F$', fontsize=25)
plt.legend(loc='best', fontsize=25)
plt.title('Mean subspace estimation error', fontsize=30)
plt.xticks(ns, fontsize=20)
plt.yticks(np.array(range(1, 8)) / 10, fontsize=20)
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
plt.grid(ls=':')
plt.savefig('figs/exp1_hypercube_value_2.png')
# plt.close()
plt.show()

