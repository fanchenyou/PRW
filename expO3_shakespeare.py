#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################################################################
# Experiment for showing PRW on learning topics in Shakespeare corpus
############################################################################
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import sys
from heapq import nlargest
sys.path.insert(0, "../")
from SRW import SubspaceRobustWasserstein
from Optimization.frankwolfe import FrankWolfe
sys.path.insert(0, "Text/")
from exp3_cinema import load_vectors, load_text

dictionnary = load_vectors('Data/Text/wiki-news-300d-1M.vec', size=20000)
dictionnary_pd = pd.DataFrame(dictionnary).T


def plot_pushforwards_wordcloud(PRW, words_X, words_Y, X, Y, a, b, corpus_name, mode_str):
    """Plot the projected measures as word clouds."""
    proj_X, proj_Y = PRW.get_projected_pushforwards()
    N_print_words = 30
    plt.figure(figsize=(10, 10))
    plt.scatter(proj_X[:, 0], proj_X[:, 1], s=X.shape[0] * 20 * a, c='r', zorder=10, alpha=0.)
    plt.scatter(proj_Y[:, 0], proj_Y[:, 1], s=Y.shape[0] * 20 * b, c='b', zorder=10, alpha=0.)
    large_a = nlargest(N_print_words, [a[words_X.index(i)] for i in words_X if i not in words_Y])[-1]
    large_b = nlargest(N_print_words, [b[words_Y.index(i)] for i in words_Y if i not in words_X])[-1]
    large_ab = \
        nlargest(N_print_words,
                 [0.5 * a[words_X.index(i)] + 0.5 * b[words_Y.index(i)] for i in words_Y if i in words_X])[
            -1]
    for i in range(a.shape[0]):
        if words_X[i] in ['thou', 'thee', 'thy']:
            continue
        if a[i] > large_a:
            if words_X[i] not in words_Y:
                plt.gca().annotate(words_X[i], proj_X[i, :], size=2500 * a[i], color='b', ha='center', alpha=0.8)
    for j in range(b.shape[0]):
        if words_Y[j] in ['thou', 'thee', 'thy']:
            continue
        if b[j] > large_b and words_Y[j] not in words_X:
            plt.gca().annotate(words_Y[j], proj_Y[j, :], size=2500 * b[j], color='r', ha='center', alpha=0.8)
        elif words_Y[j] in words_X and 0.5 * b[j] + 0.5 * a[words_X.index(words_Y[j])] > large_ab:
            size = 0.5 * b[j] + 0.5 * a[words_X.index(words_Y[j])]
            plt.gca().annotate(words_Y[j], proj_Y[j, :], size=2500 * size, color='darkviolet', ha='center', alpha=0.8)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig('./figs/exp3_%s_wordcloud_%s.png' % (corpus_name, mode_str))
    plt.title('SRW projection of word clouds', fontsize=15)

    # plt.show()
    plt.close()


#########################################################################
# Shakespeare articles
# COMPUTE SRW DISTANCES BETWEEN THE Corpus
# https://www.folgerdigitaltexts.org/download/txt.html
#########################################################################
# names = ['Henry IV, Part 1', 'Henry IV, Part 2', 'Henry V', 'Hamlet', 'Julius Caesar', 'King Lear', 'Macbeth',
#          'A Midsummer Night Dream', 'The Merchant of Venice', 'Othello', 'Romeo and Juliet', 'The Two Gentlemen of Verona']
# scripts = ['H4_1.txt', 'H4_2.txt', 'H5.txt', 'Ham.txt', 'JC.txt', 'Lr.txt', 'Mac.txt', 'MND.txt', 'MV.txt', 'Oth.txt',
#            'Rom.txt', 'TGV.txt']

# names = ['Henry V', 'Hamlet', 'Julius Caesar', 'King Lear', 'Macbeth', 'The Merchant of Venice', 'Othello',
#          'Romeo and Juliet']
# scripts = ['H5.txt', 'Ham.txt', 'JC.txt', 'Lr.txt', 'Mac.txt', 'MV.txt', 'Oth.txt', 'Rom.txt']

names = ['Henry V', 'Hamlet', 'Julius Caesar', 'The Merchant of Venice', 'Othello',
         'Romeo and Juliet']
scripts = ['H5.txt', 'Ham.txt', 'JC.txt', 'MV.txt', 'Oth.txt', 'Rom.txt']



print(len(names), len(scripts))
assert len(names) == len(scripts)

# scripts = ['H4_1.txt', 'H4_2.txt', 'H5.txt']
Nb_scripts = len(scripts)
SRW_matrix = np.zeros((Nb_scripts, Nb_scripts))
measures = []
for film in scripts:
    measures.append(load_text('Data/Text/Shakespeare/' + film))
mode_str = 'srw'


for art1 in scripts:
    for art2 in scripts:
        i = scripts.index(art1)
        j = scripts.index(art2)
        if i < j:
            X, a, words_X = measures[i]
            Y, b, words_Y = measures[j]

            algo = FrankWolfe(reg=0.2, step_size_0=None, max_iter=30, threshold=0.01, max_iter_sinkhorn=30,
                              threshold_sinkhorn=1e-3, use_gpu=False)
            SRW = SubspaceRobustWasserstein(X, Y, a, b, algo, k=2)
            SRW.run()

            SRW_matrix[i, j] = SRW.get_value()
            SRW_matrix[j, i] = SRW_matrix[i, j]
            print('SRW (', art1, ',', art2, ') =', SRW_matrix[i, j])
            if 'H5' in art1 and 'JC' in art2:
                plot_pushforwards_wordcloud(SRW, words_X, words_Y, X, Y, a, b,
                                        'shakespeare_%s_%s' % (art1[:-4], art2[:-4]), mode_str)

print()
for i in range(Nb_scripts):
    print('%s ' % (scripts[i][:-4]), end=' ')
    tmp = np.array(SRW_matrix[i, :])
    tmp[i] = 1000
    min_val = min(tmp)
    for j in range(Nb_scripts):
        if SRW_matrix[i, j] == min_val:
            print('& \\textbf{%.3f} ' % (SRW_matrix[i, j]), end='')
        else:
            print('& %.3f ' % (SRW_matrix[i, j]), end='')
    print('\\\\ \hline')
print()

# Plot the metric MDS projection of the SRW values
PRW_all = pd.DataFrame(SRW_matrix, index=scripts, columns=scripts)

embedding = MDS(n_components=2, dissimilarity='precomputed')
dis = PRW_all - PRW_all[PRW_all > 0].min().min()
dis.values[[np.arange(dis.shape[0])] * 2] = 0
embedding = embedding.fit(dis)
X = embedding.embedding_

plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], alpha=0.)
plt.axis('equal')
plt.axis('off')
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'gold', 'lime', 'plum', 'violet']
for ix, film in enumerate(scripts):
    i = scripts.index(film)
    plt.gca().annotate(film[:-4].replace('_', ' '), X[i], size=15, ha='center', color=colors[i], weight="bold")
plt.savefig('./figs/exp3_shakespeare_similarity_%s.png' % (mode_str,))
# plt.show()

# Print the most similar movie to each movie
for film in scripts:
    print('The film most similar to', film[:-4].replace('_', ' '), 'is',
          PRW_all[film].loc[PRW_all[film] > 0].idxmin()[:-4].replace('_', ' '))
