#!/usr/bin/env python 
# -*- coding: utf-8 -*-
""" 
@version: py3.5        @license: Apache Licence  
@author: 'Treamy'    @contact: chenymcan@gmail.com 
@file: logistic_regression.py      @software: PyCharm 
@time: 2018/4/24 15:01 @site: www.chenymcan.com
"""
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def sigmoid(w,x):
    z =np.dot(x,w)
    return 1./(1.+np.exp(-z))

def sigmoid_p(w,x):
    z = np.dot(x, w)
    return np.exp(-z)/((1.+ np.exp(-z))**2)

def sign(x):
    if x > 0.5:
        return 1
    else:
        return 0

class LogisticRegression():
    def __init__(self,epsilon=1e-6,max_iter=500):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.iter = 0


    def calc_hasse(self,w,X):
        n_X, n_feat = X.shape
        pi_p_z = sigmoid_p(w, X)
        H = np.zeros((n_feat,n_feat))
        for i in range(n_feat):
            for j in range(n_feat):
                temp = np.multiply(X[:,i],X[:,j])
                H[i,j] = np.dot(pi_p_z,temp)
        return H

    def train(self,X_train,y_train):
        n_X,n_feat = X_train.shape
        w = np.ones((n_feat+1,))
        X = np.hstack((np.ones((n_X,1)), X_train)) # n_X * 1+n_feat
        i = 0
        while i<self.max_iter:
            pi_z = sigmoid(w,X)
            s = y_train - pi_z
            gradient = np.dot(s,X)
            if norm(gradient)< self.epsilon:
                break
            else:
                # hasse = self.calc_hasse(w,X,)
                # lbd = np.dot(gradient,gradient) / np.dot(np.dot(gradient,hasse),gradient)
                lbd = 0.001
                w_n = w + lbd*gradient
                if norm(w_n-w)<self.epsilon:
                    break
                else:
                    w = w_n
            i = i+1
        self.w = w
        self.iter = i
        return self

    def plot2D(self,X,y):
        n = len(X)
        index1 = y==1
        index0 = y==0
        X_1 = X[index1,:]
        X_0 = X[index0,:]
        x1_s, x1_e = X[:, 0].min() - 0.5, X[:, 0].max()+0.5
        x2_s, x2_e = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(X_1[:,0], X_1[:,1], s=30, c='red', marker='s')
        ax.scatter(X_0[:,0], X_0[:,1], s=30, c='green')
        X1,X2 = np.meshgrid(np.linspace(x1_s,x1_e,50),np.linspace(x2_s,x2_e,50))
        X_ = np.vstack((X1.flatten(),X2.flatten())).T
        Y = self.prediction4plot(X_).reshape(X1.shape)

        plt.contour(X1,X2,Y,[0.5],colors="k",linewidths=1,origin="lower")
        plt.title("Logistic Regression")
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()

    def prediction4plot(self,X):
        n_X, n_feat = X.shape
        X = np.hstack((np.ones((n_X, 1)), X))
        pi_z = sigmoid(self.w, X)
        return pi_z

    def prediction(self,X):
        n_X, n_feat = X.shape
        X = np.hstack((np.ones((n_X,1)), X))
        pi_z = sigmoid(self.w, X)
        predi = np.array(list(map(sign,pi_z)))
        return predi

    def accurancy(self,X,y):
        predi = self.prediction(X)
        return sum(predi==y)/len(y)

def loadDataSet(path):
    dataMat = []; labelMat = []
    fr = open(path)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return np.array(dataMat),np.array(labelMat)



if __name__ == "__main__":
    X,y = loadDataSet("testSet.txt")
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
    lg = LogisticRegression()
    lg = lg.train(X_train,y_train)
    # print(lg.iter)
    acc = lg.accurancy(X_train,y_train)
    acc2 = lg.accurancy(X_test,y_test)
    print(acc,acc2)
    lg.plot2D(X,y)


