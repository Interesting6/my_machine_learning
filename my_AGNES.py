#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy


import numpy as np
from numpy.linalg import norm
import pandas as pd



def distMatrix(A, B):
    x2B = lambda x: np.sum((x - B) ** 2, 1) ** .5
    M = np.array(list(map(x2B, A)))
    return M

class distance(object):
    """
    implements the different distance
    """
    @staticmethod
    def dmin():
        return lambda A, B: np.min(distMatrix(A, B))

    @staticmethod
    def dmax():
        return lambda A, B: np.max(distMatrix(A, B))

    @staticmethod
    def davg():
        return lambda A, B: np.mean(distMatrix(A, B))


# class myAGNES(object):
#     def __init__(self, distfunc):
#         self.d = distfunc


dataset = pd.read_excel('watermelon_dataset.xlsx', index_col='ID')
X = dataset.values[:, :-1] # m*d
m,d = X.shape



M = distMatrix(X, X)
M[np.where(M==0.)] = np.inf

dfun = distance.dmax()

C = [point.reshape(1,d) for point in X] # at the beginning every cluster is a matrix that contain a d-dim point
q = m # current #Clusters is setup to m
k = 7 # aim to k clusters


while q > k:
    i, j = np.where(M == np.min(M))
    i, j = i[0], j[0]
    i, j  = min(i,j), max(i,j)
    C[i] = np.concatenate((C[i], C[j]), axis=0) #
    C[j:q-1] = C[j+1:q]
    del C[-1] # Notice this delete otherwise your cluster number is not reduced
    M = np.delete(M, j, axis=0)
    M = np.delete(M, j, axis=1)
    M[i, :] = np.array(list(map(dfun, [C[i]]*(q-1), C[:q-1])))
    M[:, i] = M[i, :]
    M[i, i] = np.inf # Notice this infinite otherwise you'll get the min dist of the same point
    q -= 1

# print(C)
Cresult = {} # to collect the point's index in every cluster

for i,clu in enumerate(C):
    Cresult[i] = []
    for point in clu:
        idx = np.where(np.all(X==point, axis=1))[0][0]
        Cresult[i].append(idx)


print(Cresult)
