#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy


import numpy as np
from numpy.linalg import eig
import pandas as pd
from sklearn.datasets import load_iris


datasets = load_iris()
X = datasets['data']
y = datasets['target']

# print(datasets)

def distMatrix(A, B):
    x2B = lambda x: np.sum((x - B) ** 2, 1) ** .5
    M = np.array(list(map(x2B, A)))
    return M

m, d = X.shape
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

e_vals, e_vecs = eig(B)

d_p = 2

idx = np.argsort(e_vals, ) < 2
vals = e_vals[idx]
vecs = e_vecs[idx]
X_reduction = np.dot(vals, vecs**.5)
