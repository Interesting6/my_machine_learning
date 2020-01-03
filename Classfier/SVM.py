#!/usr/bin/env python 
# -*- coding: utf-8 -*-
""" 
@version: py3.5        @license: Apache Licence  
@author: 'Treamy'    @contact: chenymcan@gmail.com 
@file: my_svm.py      @software: PyCharm 
@time: 2018/1/26 13:17 @site: www.ymchen.cn
"""

import numpy as np
import pickle
from kernel import Kernel
from _functools import reduce

MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5
toler = 1e-4  # KKT条件之一

class SVM(object):
    def __init__(self, C=1, maxIter=500, kernel_option = ("linear",0)):
        self._C = C  # 惩罚参数
        # self._toler = toler # 迭代的终止条件之一
        self._b = 0  # 阈值
        self._max_iter = maxIter  # 最大迭代次数
        self._kernel_opt = kernel_option # 选用的核函数及其参数
        self._kernel = self.get_kernel()

    def get_kernel(self):
        if self._kernel_opt[0] == "rbf":
            return Kernel.gaussian(self._kernel_opt[1])
        elif self._kernel_opt[0] == "linear":
            return Kernel.linear()
        elif self._kernel_opt[0] == "poly":
            return Kernel._polykernel(self._kernel_opt[1],self._kernel_opt[2])
        else:
            return 0

    def _calc_kernel_matrix(self, ):
        """计算核函数的矩阵
        :param train_x(matrix): 训练样本的特征值
        :param kernel_option(tuple):  核函数的类型以及参数
        :return: kernel_matrix(matrix):  样本的核函数的值
        """
        f = lambda x:np.array(list(map(self._kernel, self._X_train, [x] * self.n_samples)))
        K = np.array(list(map(f, self._X_train)))
        return K

    def train(self, dataSet, labels, ):
        # 1.输入数据集
        dataSet, labels = (dataSet.A, labels.A.flatten()) if type(dataSet)\
                == np.matrixlib.defmatrix.matrix else (dataSet, labels)
        self._X_train, self._y_train = dataSet , labels   # 训练数据集及标签
        self.n_samples = np.shape(dataSet)[0]  # 训练样本的个数
        self.alphas = np.zeros(self.n_samples )  # 拉格朗日乘子（一个全0的一维向量）
        self._error_tmp = np.zeros((self.n_samples, 2))  # 保存E的缓存
        self._kernel_mat = self._calc_kernel_matrix()  # 核函数矩阵
        # 2.开始训练
        entire_set = True
        alpha_pairs_changed = 0 # 两个alpha对改变的次数
        iteration = 0

        while iteration<self._max_iter and (alpha_pairs_changed>0 or entire_set):
            # 主要在(alpha1,alpha2)不改变时，这时遍历了所有样本，entire_et由最后一步变为False，while结束
            print("\t iteration: ",iteration)
            alpha_pairs_changed = 0

            if entire_set:  # 对所有样本
                for x in range(self.n_samples):
                    alpha_pairs_changed += self.choose_and_update(x)
                iteration += 1
            else:  # 对非边界样本
                bound_samples = []
                for i in range(self.n_samples):
                    if self.alphas[i] > 0 and self.alphas[i] < self._C:
                        bound_samples.append(i)
                for x in bound_samples:
                    alpha_pairs_changed += self.choose_and_update(x)
                iteration += 1

            if entire_set:
                entire_set = False
            elif alpha_pairs_changed == 0: # 在遍历边界样本后，alpha对没改变，则遍历所有样本
                entire_set = True

        self._support_vector_indices = self.alphas > MIN_SUPPORT_VECTOR_MULTIPLIER
        self._support_vectors_num = len(self._support_vector_indices)
        # 统一为数组形式，使得行（或列）向量为一维数组，X为二维数组
        self._support_multipliers = self.alphas[self._support_vector_indices]
        self._support_vectors = self._X_train[self._support_vector_indices]
        self._support_vector_labels = self._y_train[self._support_vector_indices]
        return self

    def choose_and_update(self, alpha_index_i):
        """判断和选择两个alpha进行更新
        :param alpha_index_i(int): 选出的第一个变量的index
        :return:
        """
        error_i = self.cal_error(alpha_index_i) # 计算第一个样本的E_i
        alpha_i,X_i,y_i = self.alphas[alpha_index_i].copy(), self._X_train[alpha_index_i].copy(), self._y_train[alpha_index_i].copy() # 深拷贝
        if (y_i*error_i<-toler and alpha_i<self._C) or (y_i*error_i>toler and alpha_i>0):
            # 第一个变量违反KKT条件
            # 1.选择第二个变量
            alpha_index_j, error_j = self.select_second_sample_j(alpha_index_i, error_i)
            alpha_j,X_j, y_j = self.alphas[alpha_index_j].copy(),self._X_train[alpha_index_j].copy(), self._y_train[alpha_index_j].copy() # 深拷贝
            alpha_i_old = alpha_i.copy()
            alpha_j_old = alpha_j.copy()
            # 2.计算上下界
            if y_i != y_j:
                L = max(0, alpha_j - alpha_i)
                H = min(self._C, self._C + alpha_j - alpha_i)
            else:
                L = max(0, alpha_j + alpha_i - self._C)
                H = min(self._C, alpha_j + alpha_i)
            if L == H:
                return 0
            # 3.计算eta
            eta = self._kernel_mat[alpha_index_i, alpha_index_i] + self._kernel_mat[alpha_index_j, alpha_index_j] - 2.0 * self._kernel_mat[alpha_index_i, alpha_index_j]
            if eta <= 0: # 因为这个eta>=0
                return 0
            # 4.更新alpha_j
            self.alphas[alpha_index_j] += y_j * (error_i - error_j) / eta
            alpha_j = self.alphas[alpha_index_j].copy()
            # 5.根据范围确实最终的alpha_j
            if alpha_j > H:
                self.alphas[alpha_index_j] = H
                # 只需要改self.alphas[alpha_index_j]，alpha_j就会跟着改变，但不能改alpha_j，否则以后不会跟着前者改变
            if alpha_j < L:
                self.alphas[alpha_index_j] = L
            alpha_j = self.alphas[alpha_index_j]
            # 6.判断是否结束
            if abs(alpha_j_old-alpha_j)<1e-5:
                self._update_error_tmp(alpha_index_j)
                return 0
            # 7.更新alpha_i
            self.alphas[alpha_index_i] += y_i * y_j * (alpha_j_old - alpha_j)
            alpha_i = self.alphas[alpha_index_i].copy()
            # 8.更新b
            b1 = self._b - error_i - y_i * self._kernel_mat[alpha_index_i, alpha_index_i] * (alpha_i - alpha_i_old) \
                 - y_j * self._kernel_mat[alpha_index_i, alpha_index_j] * (alpha_j - alpha_j_old)
            b2 = self._b - error_j - y_i * self._kernel_mat[alpha_index_i, alpha_index_j] * (alpha_i - alpha_i_old) \
                 - y_j * self._kernel_mat[alpha_index_j, alpha_index_j] * (alpha_j - alpha_j_old)
            if 0<alpha_i and alpha_i<self._C:
                self._b = b1
            elif 0<alpha_j and alpha_j<self._C:
                self._b = b2
            else:
                self._b = (b1 + b2) / 2.0
            # 9.更新error
            self._update_error_tmp(alpha_index_j)
            self._update_error_tmp(alpha_index_i)
            return 1
        else:
            return 0


    def cal_error(self, alpha_index_k):
        """误差值的计算
        :param alpha_index_k(int): 输入的alpha_k的index_k
        :return: error_k(float): alpha_k对应的误差值
        np.multiply(svm.alphas,svm.train_y).T 为一个行向量（αy,αy,αy,αy,...,αy）
        """
        tmp = np.multiply(self.alphas, self._y_train).T
        predict_k = float(np.dot(tmp, self._kernel_mat[:, alpha_index_k]) + self._b)
        error_k = predict_k - float(self._y_train[alpha_index_k])
        return error_k

    def _update_error_tmp(self, alpha_index_k):
        """重新计算误差值，并对其标记为已被优化
        :param alpha_index_k: 要计算的变量α
        :return: index为k的alpha新的误差
        """
        error = self.cal_error(alpha_index_k)
        self._error_tmp[alpha_index_k] = [1, error]

    def select_second_sample_j(self, alpha_index_i, error_i):
        """选择第二个变量
        :param alpha_index_i(float): 第一个变量alpha_i的index_i
        :param error_i(float): E_i
        :return:第二个变量alpha_j的index_j和误差值E_j
        """
        self._error_tmp[alpha_index_i] = [1, error_i] # 1用来标记已被优化
        candidate_alpha_list = np.nonzero(self._error_tmp[:, 0])[0]  # 因为是列向量，列数[1]都为0，只需记录行数[0]
        max_step,alpha_index_j,error_j = 0,0,0

        if len(candidate_alpha_list)>1:
            for alpha_index_k in candidate_alpha_list:
                if alpha_index_k == alpha_index_i:
                    continue
                error_k = self.cal_error(alpha_index_k)
                if abs(error_k-error_i)>max_step:
                    max_step = abs(error_k-error_i)
                    alpha_index_j,error_j = alpha_index_k,error_k
        else:   # 随机选择
            alpha_index_j = alpha_index_i
            while alpha_index_j == alpha_index_i:
                alpha_index_j = np.random.randint(0, self.n_samples)
            error_j = self.cal_error(alpha_index_j)
        return alpha_index_j, error_j



    def get_predict_func(self):
        return lambda x: self._b + sum(map(lambda a,b,c,d:a * b *
            self._kernel(c, d),self._support_multipliers, self._support_vector_labels,
            self._support_vectors,[x]*self._support_vectors_num))

    def predict(self, x):
        """对输入的数据预测（预测一个数据）
        :param x: 要预测的数据（一个）
        :return: 预测值
        """
        kernel_value = np.array(list(map(self._kernel, self._X_train, [x] * self.n_samples)))
        predict = np.dot(np.multiply(self._y_train, self.alphas) * kernel_value) + self._b
        return predict

    def predict_data_set(self, data_X):
        '''对样本进行预测（预测多个数据）
        input:  test_data(mat):测试数据
        output: prediction(list):预测所属的类别
        '''
        predict_f = self.get_predict_func()
        return np.array(list(map(predict_f, data_X)))

    def calc_accuracy(self, test_x, test_y):
        """
        calculates the accuracy.
        """
        n_samples = np.shape(test_x)[0]
        test_x, test_y = (test_x.A, test_y.A.flatten()) if type(test_x) \
                 == np.matrixlib.defmatrix.matrix else (test_x, test_y)
        predict_arr = self.predict_data_set(test_x)
        predict_arr = np.sign(predict_arr)
        right_arr = predict_arr == test_y
        correct = sum(right_arr)
        accuracy = correct / n_samples
        return accuracy

    def get_train_accuracy(self):
        accuracy = self.calc_accuracy(self._X_train, self._y_train)
        return accuracy

    def save_svm_model(self, model_file):
        with open(model_file, "w") as f:
            pickle.dump(self, f)

    def load_svm_model(self, model_file):
        with open(model_file, "r") as f:
            svm_model = pickle.load(f)
        return svm_model

    def save_prediction(self, result_file, prediction):
        '''保存预测的结果
        input:  result_file(string):结果保存的文件
                prediction(list):预测的结果
        '''
        f = open(result_file, 'w')
        f.write(" ".join(prediction))
        f.close()



def load_data(data_file):
    data_set, labels = [], []
    with open(data_file,"r") as f:
        textlist = f.readlines()
        for line in textlist:
            tmp = []
            line = line.strip().split(" ")
            labels.append(float(line[0]))
            i = 1
            for word in line[1:]:
                feature,value = word.split(":")
                while int(feature) != i:
                    tmp.append(float(0))
                    i += 1
                tmp.append(float(value))
                i += 1
            data_set.append(tmp)

    return (np.mat(data_set),np.mat(labels).T)


if __name__ == "__main__":
    train_x,train_y = load_data("heart_scale")
    # print(train_y,train_x)
    svm = my_svm.SVM(C=0.6,kernel_option=("rbf",0.431029))
    svm = svm.SVM_training(train_x,train_y,)
    # print(svm.alphas,svm.b)
    accuracy = svm.get_train_accracy()
    print("The training accuracy is: %.3f%%" % (accuracy * 100))
