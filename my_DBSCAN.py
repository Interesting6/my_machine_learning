#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy

import numpy as np
import pandas as pd
from random import choice




class myDBSCAN(object):
    def __init__(self, MinPts=5, epsilon = 0.11,):
        self.MinPts = MinPts
        self.epsilon = epsilon

    def get_core_obj(self, X):
        core_obj = []
        N = {}
        for xidx, xi in enumerate(X):
            d = np.sum((X - xi) ** 2, 1) ** .5
            N[xidx] = d <= self.epsilon
            if sum(N[xidx]) >= self.MinPts:
                core_obj.append(xi.tolist())
        return core_obj, N # core obj & its epsilon neighbor

    def train(self,X, ):
        self.core_obj, self.N = self.get_core_obj(X)
        tao = X.tolist() if type(X) is np.ndarray else X
        X_li = X.tolist() if type(X) is np.ndarray else X
        C = {} # cluster
        arrived = []
        k = 0
        while self.core_obj:
            tao_old = tao.copy()
            obj = choice(self.core_obj)
            queue = [obj]
            tao.remove(obj)
            while queue:
                q = queue.pop(0)
                qidx = X_li.index(q)
                if sum(self.N[qidx]) >= self.MinPts:
                    delta = [i.tolist() for i in X[self.N[qidx]] if i.tolist() in tao]
                    queue += delta
                    tao = [i for i in tao if i not in delta]
            k += 1
            C[k] = [i for i in tao_old if i not in tao]
            self.core_obj = [i for i in self.core_obj if i not in C[k] ]
            arrived += C[k]

        for key in C:
            C[key] = [(X_li.index(i), i) for i in C[key]]
        C[0] = [(X_li.index(i), i) for i in X_li if i not in arrived] # noise point
        # print(C)
        self.Cluster = C

    def predict(self, x):
        pass



if __name__ == '__main__':
    dataset = pd.read_excel('watermelon_dataset.xlsx', index_col='ID')
    X = dataset.values[:, :-1]  # m*d
    m, d = X.shape
    DBSCAN = myDBSCAN()
    DBSCAN.train(X)
    print(DBSCAN.Cluster)

