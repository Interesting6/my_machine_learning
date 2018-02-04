#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@version: py3.5        @license: Apache Licence
@author: 'Treamy'    @contact: chenymcan@gmail.com
@file: my_knn.py      @software: PyCharm
@time: 2018/2/4 22:05 @site: www.ymchen.cn
"""

import numpy as np

class my_knn(object):
    """docstring for my_knn"""
    def __init__(self, k):
        super(my_knn, self).__init__()
        self.k = k
        # self.X_test, self.y_test = np.array(X_test), np.array(y_test) X_test, y_test,

    def train(self, X_train, y_train):
        self.X_train, self.y_train = np.array(X_train), np.array(y_train)
        if len(self.X_train) != len(self.y_train):
            raise ValueError("X_test,y_test or y_train was not equail!"
                             "The length of X_test,y_test is %s"
                             "But the length of y_train is %s" % (len(self.X_train), len(self.y_train)))
        return self

    def predict_one(self, X):
        dist2xtrain = np.sum((X - self.X_train)**2, axis=1)**0.5
        index = dist2xtrain.argsort() # 从小到大（近到远）
        label_count = {}
        for i in range(self.k):
            label = self.y_train[index[i]]
            label_count[label] = label_count.get(label, 0) + 1
        # 将label_count的值从大到小排列label_count的键
        y_predict = sorted(label_count, key=lambda x: label_count[x], reverse=True)[0]
        return y_predict

    def predict_all(self, X):
        return np.array(list(map(self.predict_one, X)))

    def calc_accuracy(self, X, y):
        predict = self.predict_all(X)
        total = X.shape[0]
        right = sum(predict == y)
        accuracy = right/total
        return accuracy


def run():
    pass


if __name__ == "__main__":

    knn = my_knn(1)
    data = np.array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]])
    labels = np.array([1, 1, 1, 2, 2, 2])
    knn = knn.train(data,labels)
    accuracy = knn.calc_accuracy(data,labels)
    print("%.3f%%" % (accuracy * 100))

