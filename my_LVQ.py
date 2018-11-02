#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
# import sys

class myLVQ(object):
    def __init__(self,q=5, eta=0.1, mx_iter=500, e=1e-5):
        self.mx_iter = mx_iter ; self.e=e
        self.eta = eta
        self.q = q

    def train(self, dataset):
        pt = dataset.sample(self.q, random_state=0).values
        self.p, self.t = pt[:,:-1], pt[:,-1]
        itera = 0
        while (itera < self.mx_iter):
            xy = dataset.sample(1, random_state=0).values[0]
            x, y = xy[:-1], xy[-1]
            d = np.sum((x - self.p)**2, 1)
            idx = np.argmin(d)
            if y==self.t[idx]:
                self.p[idx] = self.p[idx] + self.eta * (x - self.p[idx])
            else:
                self.p[idx] = self.p[idx] - self.eta * (x - self.p[idx])
            itera += 1

    def predict(self, x):
        d = np.sum((x - self.p) ** 2, 1)
        idx = np.argmin(d)
        return idx

    def accuracy(self, X, y):
        pass

    def plot2D(self, dataset):
        X = dataset.values[:, :-1]
        y = dataset.values[:, -1]
        index1 = y == 1
        index2 = y == 2
        plt.scatter(X[index1, 0], X[index1, 1], marker='o',color='r')
        plt.scatter(X[index2, 0], X[index2, 1], marker='^',color='b')
        plt.show()


if __name__ == '__main__':
    dataset = pd.read_excel("/Users/treamy/code/pydir/pycharm/ML/watermelon_dataset.xlsx",index_col='ID')
    # print(dataset)
    LVQclf = myLVQ()
    LVQclf.train(dataset)
    print(LVQclf.p)
    LVQclf.plot2D(dataset)


