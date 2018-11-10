#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy


import numpy as np
from numpy.linalg import eig
import pandas as pd
from sklearn.datasets import load_iris


def distMatrix(A, B):
    x2B = lambda x: np.sum((x - B) ** 2, 1) ** .5
    M = np.array(list(map(x2B, A)))
    return M



class myMDS(object):
    def __init__(self, d_r = 2):
        self.d_r = d_r


    def train(self, X):
        D = distMatrix(X, X)
        D2 = D**2
        Di_ = np.sum(D2, 1)/m
        D_j = np.sum(D2, 0)/m
        D__ = np.sum(D2) / (m*m)

        B = np.zeros_like(D)
        for i in range(m):
            for j in range(i, m):
                B[i, j] = -0.5 * (D2[i, j] - Di_[i] - D_j[j] + D__)
                B[j, i] = B[i, j]

        # B = np.matrix(B)
        e_vals, e_vecs = eig(B)

        idx = np.argsort(-e_vals) < self.d_r
        vals = e_vals[idx] # 从大到小的d_r个特征值 dr*dr
        Lambda = np.diag(vals)
        vecs = e_vecs[:, idx] # 从大到小的d_r个特征向量(列向量) m*dr
        X_reduction = np.dot(vecs, Lambda**.5 ) # m*dr * dr*dr = m*dr
        # print(X_reduction.shape)
        # print(X_reduction)
        self.X_reduction = X_reduction


if __name__ =='__main__':
    datasets = load_iris()
    X = datasets['data']
    y = datasets['target']
    m, d = X.shape

    # print(datasets)
    # print(X)

    MDS = myMDS()
    MDS.train(X)
    print(MDS.X_reduction)



