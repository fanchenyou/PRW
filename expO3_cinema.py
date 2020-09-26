#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################################################################
# Experiment for showing SRW on learning topics on Movie scripts
############################################################################

import numpy as np
import pandas as pd
from collections import Counter
from heapq import nlargest
import matplotlib.pyplot as plt
import sys
from sklearn.manifold import MDS

sys.path.insert(0, "../")
from SRW import SubspaceRobustWasserstein
from Optimization.frankwolfe import FrankWolfe
from exp3_cinema import load_vectors, load_text, plot_pushforwards_wordcloud

sys.path.insert(0, "./Data/Text/")

dictionnary = load_vectors('Data/Text/wiki-news-300d-1M.vec', size=20000)
dictionnary_pd = pd.DataFrame(dictionnary).T

#########################################################################
# COMPUTE SRW DISTANCES BETWEEN THE MOVIES
# AND PLOT THE OPTIMAL SUBSPACE BETWEEN KILL BILL VOL.1 AND INTERSTELLAR
#########################################################################

scripts = ['DUNKIRK.txt', 'GRAVITY.txt', 'INTERSTELLAR.txt', 'KILL_BILL_VOLUME_1.txt', 'KILL_BILL_VOLUME_2.txt',
           'THE_MARTIAN.txt', 'TITANIC.txt']
abbreviations = ['D', 'G', 'I', 'KB1', 'KB2', 'TM', 'T']
Nb_scripts = len(scripts)
SRW_matrix = np.zeros((Nb_scripts, Nb_scripts))
measures = []
for film in scripts:
    measures.append(load_text('Data/Text/Cinema/' + film))

np.random.seed(357)
mode_str = 'srw'

for film1 in scripts:
    for film2 in scripts:
        i = scripts.index(film1)
        j = scripts.index(film2)
        # if film2 != 'KILL_BILL_VOLUME_1.txt' or film1 != 'INTERSTELLAR.txt':
        #     continue

        if i < j:
            X, a, words_X = measures[i]
            Y, b, words_Y = measures[j]
            algo = FrankWolfe(reg=0.1, step_size_0=None, max_iter=50, threshold=0.01, max_iter_sinkhorn=30,
                              threshold_sinkhorn=1e-3, use_gpu=False)
            SRW = SubspaceRobustWasserstein(X, Y, a, b, algo, k=2)
            SRW.run()
            # SRW.plot_convergence()
            SRW_matrix[i, j] = SRW.get_value()
            SRW_matrix[j, i] = SRW_matrix[i, j]
            print('SRW (', film1, ',', film2, ') =', SRW_matrix[i, j])
            if film2 == 'KILL_BILL_VOLUME_1.txt' and film1 == 'INTERSTELLAR.txt':
                plot_pushforwards_wordcloud(SRW, words_X, words_Y, X, Y, a, b,
                                        'cinema_%s_%s' % (film1[:-4], film2[:-4]), mode_str)

# print latex scripts for table
print()
for i in range(Nb_scripts):
    print('%s ' % (abbreviations[i]), end=' ')
    for j in range(Nb_scripts):
        print('& %.3f ' % (SRW_matrix[i, j]), end='')
    print('\\\\ \hline')
print()

# Plot the metric MDS projection of the SRW values
SRW_all = pd.DataFrame(SRW_matrix, index=scripts, columns=scripts)

embedding = MDS(n_components=2, dissimilarity='precomputed')
dis = SRW_all - SRW_all[SRW_all > 0].min().min()
dis.values[[np.arange(dis.shape[0])] * 2] = 0
embedding = embedding.fit(dis)
X = embedding.embedding_

plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], alpha=0.)
plt.axis('equal')
plt.axis('off')
c = {'KILL_BILL_VOLUME_1.txt': 'red', 'KILL_BILL_VOLUME_2.txt': 'red', 'TITANIC.txt': 'blue', 'DUNKIRK.txt': 'blue',
     'GRAVITY.txt': 'black', 'INTERSTELLAR.txt': 'black', 'THE_MARTIAN.txt': 'black'}
for film in scripts:
    i = scripts.index(film)
    plt.gca().annotate(film[:-4].replace('_', ' '), X[i], size=35, ha='center', color=c[film], weight="bold")
plt.savefig('./figs/exp3_cinema_similarity_%s.png' % (mode_str,))

# Print the most similar movie to each movie
for film in scripts:
    print('The film most similar to', film[:-4].replace('_', ' '), 'is',
          SRW_all[film].loc[SRW_all[film] > 0].idxmin()[:-4].replace('_', ' '))
