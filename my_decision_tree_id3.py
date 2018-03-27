#!/usr/bin/env python 
# -*- coding: utf-8 -*-
""" 
@version: py3.5        @license: Apache Licence  
@author: 'Treamy'    @contact: chenymcan@gmail.com 
@file: ID3.py      @software: PyCharm 
@time: 2018/3/26 18:19 @site: www.chenymcan.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

decision_node={"boxstyle": "sawtooth", "fc": "0.8", }
leaf_node={"boxstyle": "round4", "fc": "0.8"}
arrow_args={"arrowstyle": "<-"}

def run():
    pass

class ID3_tree(object):
    def __init__(self, df):
        self.df = df.copy()
        self.feat, self.cate = df.columns[:-1], df.columns[-1]

    def calc_entropy(self, P_):
        return sum(map(lambda p: -p * np.log2(p), P_))

    def train(self,df):
        df = df.copy()
        feat, cate = df.columns[:-1].tolist(), df.columns[-1]
        P_cate = df[cate].value_counts() / df[cate].count()
        self.H_D = self.calc_entropy(P_cate)
        self.my_tree = self.create_tree(df, )
        return self

    def create_tree(self, df, ):
        feat = df.columns[:-1].tolist()
        cate_values = df[self.cate].unique()
        if len(cate_values)==1:
            return cate_values[0]
        if len(feat) == 0: # 用完所有特征后
            temp = df[self.cate].value_counts().to_dict()
            return max(temp, key=lambda x:temp[x]) # 取最多的类别作为返回值
        best_feat = self.select(df, feat)
        my_tree = { best_feat:{} }
        unique_feat_values = df[best_feat].unique()
        for feat_value in unique_feat_values:
            # df_ = df[df[best_feat] == feat_value].copy()
            # df_ =  df_.drop(best_feat, axis=1)
            df_ = self.split_df(df, best_feat, feat_value)
            # 这里一定不能是df，必须是一个新的df_，才能使递归*中的feat*越来越小
            my_tree[best_feat][feat_value] = self.create_tree(df_, )
        return my_tree

    def split_df(self, df, feat, feat_value):
        df = df[df[feat] == feat_value].copy()
        return df.drop(feat, axis=1) # 删除已选取的特征列


    def select(self,df, features ):
        gain_dict = {}
        for feat in features:
            groups = df.groupby(feat)
            feat_value_entropy = 0
            for feat_value, group in groups:
                feat_value_P = len(group) / len(df)  # 特征取值的概率
                feat_value_cate_P = group[self.cate].value_counts() / group[self.cate].count()  # 特征取某值对应不同的类别的概率
                feat_value_entropy += feat_value_P * self.calc_entropy(feat_value_cate_P)
            imfor_gain = self.H_D - feat_value_entropy
            gain_dict[feat] = imfor_gain
        return max(gain_dict, key=lambda x:gain_dict[x])

    def get_leafs_num(self, tree_):
        num_leafs = 0
        first_key = list(tree_.keys())[0]
        second_dict = tree_[first_key]
        for key in second_dict.keys():
            if type(second_dict[key]).__name__ == "dict":
                num_leafs += self.get_leafs_num(second_dict[key])
            else:
                num_leafs += 1
        return num_leafs

    def get_tree_depth(self,tree_):
        max_depth = 0
        first_key = list(tree_.keys())[0]
        second_dict = tree_[first_key]
        for key in second_dict.keys():
            if type(second_dict[key]).__name__ == "dict": # 如果还是字典，继续深入
                this_depth = 1 + self.get_tree_depth(second_dict[key])
            else:
                this_depth = 1
            if this_depth > max_depth:
                max_depth = this_depth
        return max_depth


    def plot_node(self, node_txt, centerPt, parentPt, node_type):
        self.ax1.annotate(node_txt, xy=parentPt, xycoords='axes fraction',xytext=centerPt,
            textcoords='axes fraction',va="center", ha="center", bbox=node_type, arrowprops=arrow_args)

    def plot_mid_text(self, cntrPt, parentPt, txt_string):  # 在两个节点之间的线上写上字
        xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
        yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
        self.ax1.text(xMid, yMid, txt_string)  # text() 的使用

    def plot_tree(self, tree_, parent_point, node_txt):
        num_leafs = self.get_leafs_num(tree_)
        depth = self.get_tree_depth(tree_)
        first_key = list(tree_.keys())[0]
        center_point = (self.xOff + (1.0 + float(num_leafs)) / 2.0 / self.totalW, self.yOff)
        self.plot_mid_text( center_point, parent_point, node_txt)  # 在父子节点间填充文本信息
        self.plot_node(first_key, center_point, parent_point, decision_node)  # 绘制带箭头的注解
        second_dict = tree_[first_key]
        self.yOff = self.yOff - 1.0 / self.totalD
        for key in second_dict.keys():
            if type(second_dict[key]).__name__ == 'dict':  # 判断是不是字典，
                self.plot_tree(second_dict[key], center_point, str(key))  # 递归绘制树形图
            else:  # 如果是叶节点
                self.xOff = self.xOff + 1.0 / self.totalW
                self.plot_node(second_dict[key], (self.xOff, self.yOff), center_point, leaf_node)
                self.plot_mid_text((self.xOff, self.yOff), center_point, str(key))
        yOff = self.yOff + 1.0 / self.totalD

    def show_tree(self, tree_):
        fig = plt.figure(1, facecolor='white')
        fig.clf()  # 清空绘图区
        axprops = dict(xticks=[], yticks=[])
        self.ax1 = plt.subplot(111, frameon=False, **axprops)
        self.totalW = float(self.get_leafs_num(tree_))
        self.totalD = float(self.get_tree_depth(tree_))
        self.xOff = -0.5 / self.totalW  # 追踪已经绘制的节点位置 初始值为 将总宽度平分 在取第一个的一半
        self.yOff = 1.0
        self.plot_tree(tree_, (0.5, 1.0), '')  # 调用函数，并指出根节点源坐标
        plt.show()



if __name__ == "__main__":
    df = pd.read_excel("TreeData.xlsx", index_col="id")
    id3_tree = ID3_tree(df)
    id3_tree = id3_tree.train(df)
    tree_ = id3_tree.my_tree.copy()
    print(tree_)

    # print(id3_tree.get_leafs_num(tree_))
    # print(id3_tree.get_tree_depth(tree_))

    id3_tree.show_tree(tree_)


