#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
# import sys

dataset = pd.read_excel("/Users/treamy/code/pydir/pycharm/ML/watermelon_dataset.xlsx",index_col='ID')
# print(dataset)

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
            xy = dataset.sample(1).values[0]
            # while (xy in pt):
            #     xy = dataset.sample(1).values[0]
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

    def get_tar(self, X):
        return np.array(list(map(self.predict, X)))


    def plot2D(self, dataset):
        X = dataset.values[:, :-1]
        y = dataset.values[:, -1]
        self.Xt = self.get_tar(X)

        index1 = y == 1
        index2 = y == 2
        plt.scatter(X[index1, 0], X[index1, 1], marker='o',color='r')
        plt.scatter(X[index2, 0], X[index2, 1], marker='^',color='b')
        plt.scatter(self.p[:, 0], self.p[:, 1], marker='*', color='y')
        plt.show()

def make_meshgrid(x, y, stepnum=50, h=.05):
    # h_x = (x.max() - x.min())/stepnum
    # h_y = (x.max() - x.min())/stepnum
    x_min, x_max = x.min() - h, x.max() + h
    y_min, y_max = y.min() - h, y.max() + h
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, stepnum))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.get_tar(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

if __name__ == '__main__':
    LVQclf = myLVQ()
    LVQclf.train(dataset)
    print(LVQclf.p)
    # LVQclf.plot2D(dataset)
    
    X = dataset.values[:, :-1]
    y = dataset.values[:, -1]
    y_p = LVQclf.get_tar(X)
    print(y_p)
    
    X0, X1 = X[:,0], X[:,1]
    x0, y1 = make_meshgrid(X0, X1)
    plot_contours(plt, LVQclf, x0, y1,cmap=plt.cm.coolwarm, )
    plt.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.show()

