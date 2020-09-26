#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################
# Experiment to visualize OPT plan under different numbers
###############################################################
import numpy as np

from SRW import SubspaceRobustWasserstein
from Optimization.projectedascent import ProjectedGradientAscent

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
ns = [100, 250]  # Compute SRW between measures with 'n' points for 'n' in 'ns'


proj = np.zeros((d, d))  # Real optimal subspace
proj[0, 0] = 1
proj[1, 1] = 1

np.random.seed(357)


for indn in range(len(ns)):
    n = ns[indn]

    a, b, X, Y = fragmented_hypercube(n, d, dim=2)

    # Compute SRW
    algo = ProjectedGradientAscent(reg=0, step_size_0=0.01, max_iter=15, max_iter_sinkhorn=30,
                                   threshold=0.05, threshold_sinkhorn=1e-04, use_gpu=False)
    SRW_ = SubspaceRobustWasserstein(X, Y, a, b, algo, k=k)
    SRW_.run()
    SRW_.plot_transport_plan(path='figs/exp1_plan_%s_%d.png' % ('SRW', n),
                             method_name='SRW')

    # Compute Wasserstein
    algo = ProjectedGradientAscent(reg=0, step_size_0=0.01, max_iter=1, max_iter_sinkhorn=30,
                                   threshold=0.05, threshold_sinkhorn=1e-04, use_gpu=False)
    W_ = SubspaceRobustWasserstein(X, Y, a, b, algo, k=d)
    W_.run()
    W_.plot_transport_plan(path='figs/exp1_plan_%s_%d.png' % ('W', n),
                           method_name='W')

    # Riemann Gradient
    algo = RiemmanAdaptive(reg=0.1, step_size_0=None, max_iter=30, threshold=0.01,
                           max_iter_sinkhorn=30,
                           threshold_sinkhorn=1e-04, use_gpu=False)
    PRW = ProjectionRobustWasserstein(X, Y, a, b, algo, k)
    #PRW.run(0, lr=0.2, beta=None)
    PRW.run(1, lr=0.01, beta=0.8)
    PRW.plot_transport_plan('figs/exp1_plan_%s_%d.png' % ('PRW', n))

