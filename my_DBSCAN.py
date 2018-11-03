#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy

import numpy as np
import pandas as pd
from random import choice

dataset = pd.read_excel('watermelon_dataset.xlsx', index_col='ID')
X = dataset.values[:, :-1] # m*d
m,d = X.shape

MinPts = 5
epsilon = 0.11

core_obj = []
N = {}
for xidx,xi in enumerate(X):
    d = np.sum((X-xi)**2, 1) ** .5
    N[xidx] = d <= epsilon
    if sum(N[xidx]) >= MinPts:
        core_obj.append(xi.tolist())

k = 0
tao = X.tolist()
X_li = X.tolist()
C = {}
arrived = []
while core_obj:
    tao_old = tao.copy()
    obj = choice(core_obj)
    queue = [obj]
    tao.remove(obj)
    while queue:
        q = queue.pop(0)
        qidx = X_li.index(q)
        if sum(N[qidx]) >= MinPts:
            delta = [i.tolist() for i in X[N[qidx]] if i.tolist() in tao]
            queue += delta
            tao = [i for i in tao if i not in delta]
    k += 1
    C[k] = [i for i in tao_old if i not in tao]
    core_obj = [i for i in core_obj if i not in C[k] ]
    arrived += C[k]

# print(C)
for key in C:
    C[key] = [(X_li.index(i), i) for i in C[key]]
C[0] = [(X_li.index(i), i) for i in X_li if i not in arrived] # noise point
print(C)


