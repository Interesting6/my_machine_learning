#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy

"""
    SSF尺度空间聚类
"""

import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from sklearn.datasets import make_blobs
from PIL import Image




def ssf(X, fst_sig=0.5, K=4, epsilon=1e-6):
    sigs = [fst_sig*1.029**i for i in range(10)]
    M, n = X.shape
    X_temp = X
    y = np.zeros_like(X) # 用来存储每个X的收敛点
    labels = np.ones(M, dtype=int)
    m = len(X_temp)
    cluster_ids = np.array([[i] for i in range(m)])
    num = 0
    for sig in sigs:
        num += 1
        for i in range(0, m):
            x0 = X_temp[i]
            while True:
                dif = x0 - X_temp
                norm_dif = np.array([norm(dif_i) for dif_i in dif])
                tp1 = np.exp(-norm_dif/(2*sig**2))
                tp2 = np.sum(tp1)
                tp3 = np.repeat(tp1, n).reshape(-1, n)
                tp4 = np.sum(np.multiply(tp3, X_temp), axis=0)
                x1 = tp4 / tp2
                if norm(x1-x0)<epsilon:
                    y[i] = x1
                    break
                else:
                    x0 = x1
        y_uni = np.unique(y, axis=0)
        m = len(y_uni)
        cluster_list_ = [[] for _ in range(m)]

        for k, y_uni_r in enumerate(y_uni): # 对本次聚类中聚合的点合并成新簇
            idxs = np.where((y == y_uni_r[None]).all(-1))[0] # 选出本次聚类中收敛到同一点的点
            idxs_t = cluster_ids[idxs]  # 选出本次聚到同一点的上一次聚类中的原始点

            for idxs_t_t in idxs_t: # 将聚到同一点的原始点合并到一起
                cluster_list_[k].extend(idxs_t_t)
        cluster_ids = np.array([ tt for tt in cluster_list_]) # 储存为本次各个簇的原始点
        # print(cluster_list_)

        if m <= K:
            for k, r in enumerate(cluster_list_):
                labels[r] = k
            break
        else: # continue
            X_temp = y_uni
            y = np.zeros_like(X_temp)


    print('sigma iter num:', num)
    print('cluster num:', len(cluster_list_))
    return labels, y_uni


def test1():
    X, y_true = make_blobs(n_samples=100, centers=4,
                           cluster_std=0.60, random_state=0)

    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()
    # M, n = X.shape
    X = torch.tensor(X, dtype=torch.float).cuda()
    labels, y_uni = ssf(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
    plt.show()

def test2():
    """
        通过聚类压缩图片
    """
    img_path = "C:/Users/Tream/Pictures/lenna_eye.png"
    pic = Image.open(img_path)
    img = np.array(pic)
    img = img.astype('float32') / 255.
    W, D, C = img.shape

    K = 30*30
    sig = 0.1

    X = img.reshape(-1, C)
    X_temp = X

    labels, y_uni = ssf(X, fst_sig=sig, K=K)

    X_new = y_uni[labels]*255
    X_cp = X_new.reshape(W, D, C).astype(np.uint8)
    plt.imshow(X_cp)
    plt.show()



if __name__ == '__main__':
    test1()

    test2()
