#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal as mnor
import matplotlib.pyplot as plt


dataset = pd.read_excel('watermelon_dataset.xlsx', index_col='ID')
X = dataset.values[:, :-1] # m*d
m,d = X.shape
# print(X.shape)

# initialize
mx_iter = 50
itera = 0
k = 3
alpha = np.array([1/3,] * k) # 1*k
mu = np.array([X[5], X[21], X[26],]) # k*d
Sigma = np.array([np.array([[.1, 0,], [0, .1]])] * k)
# print(alpha)
# print(mu)
# print(Sigma)
gamma = np.zeros((m, k)) # m*k
while itera< mx_iter:
    for j,x_j in enumerate(X):
        # temp = np.array([norm(mu[l], Sigma[l]).pdf(x_j) for l in range(k)]) # 1*k
        # prob_x_j = np.dot(alpha, temp)  # 1*1
        # gamma[j] = alpha * temp / prob_x_j # 1*k/1 = 1*k
        temp = np.array([alpha[l]*mnor(mu[l], Sigma[l]).pdf(x_j) for l in range(k)]) # 1*k
        gamma[j] = temp / sum(temp) # 1*k/1 = 1*k

    gamma_sum_j = np.sum(gamma, 0) # 1*k
    mu = np.dot(X.T, gamma) / gamma_sum_j # d*k ./ 1*k = d*k
    mu = mu.T # k*d

    # t = X.reshape((m, 1, d)) - mu # m*1*d - k*d = m*k*d
    # # t = np.sum(t**2, 2) #在第三维(最里面层)相加得 m*k
    # # Sigma = np.dot(t.T, gamma)  / gamma_sum_j
    # t = np.array([np.dot(i.T, i) for i in t])
    for i in range(k):
        temp = np.sum([gamma[j,i]*np.outer(*[x_j-mu[i]]*2) for j,x_j in enumerate(X)], 0)
        Sigma[i] =  temp / gamma_sum_j[i]
    alpha = gamma_sum_j / m # 1*k

    itera+=1

# print(gamma)

C = {}
lbd = np.argmax(gamma, 1)
print(lbd)

for i in range(k):
    idx = lbd == i
    C[i] = X[idx]
    plt.scatter(X[idx, 0], X[idx, 1],)

plt.scatter(mu[:, 0], mu[:, 1], marker='x')
plt.show()
