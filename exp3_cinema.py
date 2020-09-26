#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################################################################
# Experiment for showing PRW on learning topics on Movie scripts
############################################################################
import os
import numpy as np
import pandas as pd
from collections import Counter
from heapq import nlargest
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "../")
from PRW import ProjectionRobustWasserstein
from Optimization.riemann_adap import RiemmanAdaptive

sys.path.insert(0, "./Data/Text/")

##################################################
# Load the Word2Vec from
# https://fasttext.cc/docs/en/english-vectors.html
##################################################
import io


def load_vectors(fname, size=None):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    i = 0
    for line in fin:
        if size and i >= size:
            break
        if i >= 2000:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array(tokens[1:], dtype='f8')
        i += 1
    return data

if not os.path.isfile('Data/Text/wiki-news-300d-1M.vec'):
    print('[Warning]')
    print('Please download word vector at https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip')
    print('Put the unzipped "wiki-news-300d-1M.vec" in Data/Text folder, then try again')
    exit()

dictionnary = load_vectors('Data/Text/wiki-news-300d-1M.vec', size=20000)
dictionnary_pd = pd.DataFrame(dictionnary).T

##################################
# Text Preprocessing
# And transformation into measures
##################################
import string


def textToMeasure(text):
    text = text.replace('\n', '')
    text = text.replace('\t', '')
    words = text.split(' ')
    table = str.maketrans('', '', string.punctuation.replace("'", ""))
    words = [w.translate(table) for w in words if len(w) > 0]
    words = [w for w in words if w in dictionnary.keys()]
    words = [w for w in words if not w[0].isupper()]
    words = [w for w in words if not w.isdigit()]
    size = len(words)
    cX = Counter(words)
    words = list(set(words))
    a = np.array([cX[w] for w in words]) / size
    X = np.array([dictionnary[w] for w in words])
    return X, a, words


def load_text(file):
    """return X,a,words"""
    with open(file) as fp:
        text = fp.read()
    return textToMeasure(text)


############
# Plotting #
############
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
        if a[i] > large_a:
            if words_X[i] not in words_Y:
                plt.gca().annotate(words_X[i], proj_X[i, :], size=2500 * min(0.01, a[i]), color='b', ha='center',
                                   alpha=0.8)
    for j in range(b.shape[0]):
        if b[j] > large_b and words_Y[j] not in words_X:
            plt.gca().annotate(words_Y[j], proj_Y[j, :], size=2500 * b[j], color='r', ha='center', alpha=0.8)
        elif words_Y[j] in words_X and 0.5 * b[j] + 0.5 * a[words_X.index(words_Y[j])] > large_ab:
            size = 0.5 * b[j] + 0.5 * a[words_X.index(words_Y[j])]
            plt.gca().annotate(words_Y[j], proj_Y[j, :], size=2500 * size, color='darkviolet', ha='center', alpha=0.8)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig('./figs/exp3_%s_wordcloud_%s.png' % (corpus_name, mode_str))
    # plt.show()
    plt.close()


#########################################################################
# computer PRW distances between texts
#########################################################################
scripts = ['DUNKIRK.txt', 'GRAVITY.txt', 'INTERSTELLAR.txt', 'KILL_BILL_VOLUME_1.txt', 'KILL_BILL_VOLUME_2.txt',
           'THE_MARTIAN.txt', 'TITANIC.txt']
abbreviations = ['D', 'G', 'I', 'KB1', 'KB2', 'TM', 'T']
Nb_scripts = len(scripts)
PRW_matrix = np.zeros((Nb_scripts, Nb_scripts))
measures = []
for film in scripts:
    measures.append(load_text('Data/Text/Cinema/' + film))

np.random.seed(357)


def main():
    mode_str = 'RAGDS'

    k = 2
    reg = 0.1
    lr = 0.08
    beta = 0.9

    for film1 in scripts:
        for film2 in scripts:
            i = scripts.index(film1)
            j = scripts.index(film2)

            if i < j:
                X, a, words_X = measures[i]
                Y, b, words_Y = measures[j]

                algo = RiemmanAdaptive(reg=reg, step_size_0=None, max_iter=50, threshold=0.01,
                                       max_iter_sinkhorn=30,
                                       threshold_sinkhorn=1e-3, use_gpu=False)

                PRW = ProjectionRobustWasserstein(X, Y, a, b, algo, k)
                PRW.run(1, lr=lr, beta=beta)

                PRW_matrix[i, j] = PRW.get_value()
                PRW_matrix[j, i] = PRW_matrix[i, j]
                print('PRW (', film1, ',', film2, ') =', PRW_matrix[i, j])
                if film1 == 'DUNKIRK.txt' and film2 == 'INTERSTELLAR.txt':
                    plot_pushforwards_wordcloud(PRW, words_X, words_Y, X, Y, a, b,
                                                'cinema_%s_%s' % (film1[:-4], film2[:-4]), mode_str)

    # print latex scripts for table
    print()
    for i in range(Nb_scripts):
        print('%s ' % (abbreviations[i]), end=' ')
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
    for film in scripts:
        print('The film most similar to', film[:-4].replace('_', ' '), 'is',
              PRW_all[film].loc[PRW_all[film] > 0].idxmin()[:-4].replace('_', ' '))


if __name__ == '__main__':
    main()
