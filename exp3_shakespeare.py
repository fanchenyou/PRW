#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################################################################
# Experiment for showing PRW on learning topics on Shakespeare corpus
############################################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from heapq import nlargest

sys.path.insert(0, "../")
from PRW import ProjectionRobustWasserstein
from Optimization.riemann_adap import RiemmanAdaptive

sys.path.insert(0, "./Data/Text/")
from exp3_cinema import load_vectors, load_text, plot_pushforwards_wordcloud

if not os.path.isfile('Data/Text/wiki-news-300d-1M.vec'):
    print('[Warning]')
    print('Please download word vector at https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip')
    print('Put the unzipped "wiki-news-300d-1M.vec" in Data/Text folder, then try again')
    exit()

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
    plt.title('PRW projection of word clouds', fontsize=15)
    plt.close()


#########################################################################
# Shakespeare plays are downloadable from
# https://www.folgerdigitaltexts.org/download/txt.html
# we pre-downloaded them in Data/Text/Shakespeare
#########################################################################
names = ['Henry V', 'Hamlet', 'Julius Caesar', 'The Merchant of Venice', 'Othello',
         'Romeo and Juliet']
scripts = ['H5.txt', 'Ham.txt', 'JC.txt', 'MV.txt', 'Oth.txt', 'Rom.txt']

# print(len(names), len(scripts))
assert len(names) == len(scripts)

Nb_scripts = len(scripts)
PRW_matrix = np.zeros((Nb_scripts, Nb_scripts))
measures = []
for art in scripts:
    measures.append(load_text('Data/Text/Shakespeare/' + art))

np.random.seed(357)


def main():
    mode_str = 'RAGAS'

    k = 2
    reg = 0.2
    lr = 0.08
    beta = 0.9

    for art1 in scripts:
        for art2 in scripts:
            i = scripts.index(art1)
            j = scripts.index(art2)
            if i < j:
                X, a, words_X = measures[i]
                Y, b, words_Y = measures[j]

                algo = RiemmanAdaptive(reg=reg, step_size_0=None, max_iter=30, threshold=0.01,
                                       max_iter_sinkhorn=30, threshold_sinkhorn=1e-3, use_gpu=False)

                PRW = ProjectionRobustWasserstein(X, Y, a, b, algo, k)
                PRW.run(1, lr=lr, beta=beta)

                PRW_matrix[i, j] = PRW.get_value()
                PRW_matrix[j, i] = PRW_matrix[i, j]
                print('PRW (', art1, ',', art2, ') =', PRW_matrix[i, j])
                if 'JC' in art1 and 'MV' in art2:
                    plot_pushforwards_wordcloud(PRW, words_X, words_Y, X, Y, a, b,
                                                'shakespeare_%s_%s' % (art1[:-4], art2[:-4]), mode_str)

    # print latex table of the PRW distances
    print()
    for i in range(Nb_scripts):
        print('%s ' % (scripts[i][:-4]), end=' ')
        tmp = np.array(PRW_matrix[i, :])
        tmp[i] = 1000
        min_val = min(tmp)
        for j in range(Nb_scripts):
            if PRW_matrix[i, j] == min_val:
                print('& \\textbf{%.3f} ' % (PRW_matrix[i, j]), end='')
            else:
                print('& %.3f ' % (PRW_matrix[i, j]), end='')
        print('\\\\ \hline')
    print()

    PRW_all = pd.DataFrame(PRW_matrix, index=scripts, columns=scripts)
    # Print the most similar movie to each movie
    for art in scripts:
        print('The art most similar to', art[:-4].replace('_', ' '), 'is',
              PRW_all[art].loc[PRW_all[art] > 0].idxmin()[:-4].replace('_', ' '))


if __name__ == '__main__':
    main()
