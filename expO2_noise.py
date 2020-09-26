# -*- coding: utf-8 -*-

####################################################################
# Experiment for testing noise errors with different dimension
####################################################################

import numpy as np
import matplotlib.pyplot as plt
from SRW import SubspaceRobustWasserstein
from Optimization.frankwolfe import FrankWolfe
import pickle

noise_level = 1
d = 20  # Total dimension
n = 100  # Number of points for each measure
l = 5  # Dimension of Wishart
reg = 6
nb_exp = 50  # Number of experiments
k = list(range(1, d + 1))  # Compute SRW for all dimension parameter k

# Save the values
no_noise = np.zeros((2, nb_exp, d))
noise = np.zeros((2, nb_exp, d))


np.random.seed(321)

if 1==2:
    for t in range(nb_exp):  # Fore each experiment
        print(t)

        a = (1. / n) * np.ones(n)
        b = (1. / n) * np.ones(n)

        mean_1 = 0. * np.random.randn(d)
        mean_2 = 0. * np.random.randn(d)

        cov_1 = np.random.randn(d, l)
        cov_1 = cov_1.dot(cov_1.T)
        cov_2 = np.random.randn(d, l)
        cov_2 = cov_2.dot(cov_2.T)

        # Draw measures
        X = np.random.multivariate_normal(mean_1, cov_1, size=n)
        Y = np.random.multivariate_normal(mean_2, cov_2, size=n)

        # Add noise
        Xe = X + noise_level * np.random.randn(n, d)
        Ye = Y + noise_level * np.random.randn(n, d)

        # Choice of step size
        ones = np.ones((n, n))
        C = np.diag(np.diag(Xe.dot(Xe.T))).dot(ones) + ones.dot(np.diag(np.diag(Ye.dot(Ye.T)))) - 2 * Xe.dot(Ye.T)
        step_size_0 = 1. / np.max(C)


        # Compute PRW(Grad) without/with noise
        vals = []
        for ki in range(1, d + 1):
            algo = FrankWolfe(reg=reg, step_size_0=step_size_0, max_iter=100, max_iter_sinkhorn=50, threshold=1e-3, threshold_sinkhorn=1e-04, use_gpu=False)
            SRW = SubspaceRobustWasserstein(X, Y, a, b, algo, k=ki)
            SRW.run()
            vals.append(SRW.get_value())
        no_noise[0, t, :] = np.sort(vals)


        vals = []
        for ki in range(1, d + 1):
            algo = FrankWolfe(reg=reg, step_size_0=step_size_0, max_iter=100, max_iter_sinkhorn=50, threshold=1e-3, threshold_sinkhorn=1e-04, use_gpu=False)
            SRW = SubspaceRobustWasserstein(Xe, Ye, a, b, algo, k=ki)
            SRW.run()
            vals.append(SRW.get_value())
        noise[0, t, :] = np.sort(vals)


        no_noise[0, t, :] /= no_noise[0, t, (d - 1)]
        noise[0, t, :] /= noise[0, t, (d - 1)]


    with open('./results/exp2_noise_srw.pkl', 'wb') as f:
        pickle.dump([no_noise, noise], f)

else:

    with open('./results/exp2_noise_srw.pkl', 'rb') as f:
        no_noise, noise = pickle.load(f)


#captions = ['PRW (RGD)', 'PRW (RAGD)']
captions = ['SRW']

for t in range(1):
    plt.figure(figsize=(17, 8))

    no_noise_t = no_noise[t,:,:]
    no_noise_mean = np.mean(no_noise_t, axis=0)
    no_noise_min = np.min(no_noise_t, axis=0)
    no_noise_10 = np.percentile(no_noise_t, 10, axis=0)
    no_noise_25 = np.percentile(no_noise_t, 25, axis=0)
    no_noise_75 = np.percentile(no_noise_t, 75, axis=0)
    no_noise_90 = np.percentile(no_noise_t, 90, axis=0)
    no_noise_max = np.max(no_noise_t, axis=0)

    noise_t = noise[t, :, :]
    noise_mean = np.mean(noise_t, axis=0)
    noise_min = np.min(noise_t, axis=0)
    noise_10 = np.percentile(noise_t, 10, axis=0)
    noise_25 = np.percentile(noise_t, 25, axis=0)
    noise_75 = np.percentile(noise_t, 75, axis=0)
    noise_90 = np.percentile(noise_t, 90, axis=0)
    noise_max = np.max(noise_t, axis=0)


    plotnonoise, = plt.plot(range(d), no_noise_mean, 'C1', label='Without Noise', lw=6)
    col_nonoise = plotnonoise.get_color()
    plt.fill_between(range(d), no_noise_25, no_noise_75, facecolor=col_nonoise, alpha=0.3)
    plt.fill_between(range(d), no_noise_10, no_noise_90, facecolor=col_nonoise, alpha=0.2)
    plt.fill_between(range(d), no_noise_min, no_noise_max, facecolor=col_nonoise, alpha=0.15)

    plotnoise, = plt.plot(range(d), noise_mean, 'C2', label='With Noise', lw=6)
    col_noise = plotnoise.get_color()
    plt.fill_between(range(d), noise_25, noise_75, facecolor=col_noise, alpha=0.3)
    plt.fill_between(range(d), noise_10, noise_90, facecolor=col_noise, alpha=0.2)
    plt.fill_between(range(d), noise_min, noise_max, facecolor=col_noise, alpha=0.15)

    plt.xlabel('Dimension', fontsize=25)
    plt.ylabel('Normalized %s value' % (captions[t]), fontsize=25)
    plt.legend(loc='best', fontsize=20)

    plt.yticks(fontsize=20)
    plt.xticks(range(d), range(1, d + 1), fontsize=20)
    plt.ylim(0.1)

    plt.legend(loc='best', fontsize=25)
    plt.title('%s distance with different dimensions' % (captions[t],), fontsize=30)
    plt.grid(ls=':')
    plt.savefig('figs/exp2_noise_srw.png')
    plt.close()
    plt.clf()

