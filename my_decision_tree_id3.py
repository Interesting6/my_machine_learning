#!/usr/bin/env python 
# -*- coding: utf-8 -*-
""" 
@version: py3.5        @license: Apache Licence  
@author: 'Treamy'    @contact: chenymcan@gmail.com 
@file: ID3.py      @software: PyCharm 
@time: 2018/3/26 18:19 @site: www.ymchen.cn
"""

import numpy as np
import pandas as pd


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
        if len(feat) == 0:
            return self.marjority(df[self.cate])
        best_feat = self.select(df, feat)
        my_tree = { best_feat:{} }
        unique_feat_values = df[best_feat].unique()
        # df = df.drop(best_feat, axis=1)
        for feat_value in unique_feat_values:
            df_ = self.split_df(df,best_feat,feat_value)
            # 这里一定不能是df，必须是一个新的df_，才能使递归*中的feat*越来越小
            my_tree[best_feat][feat_value] = self.create_tree(df_, )
        return my_tree

    def split_df(self, df, feat, feat_value):
        df = df[df[feat] == feat_value].copy()
        return df.drop(feat, axis=1)

    def marjority(self,se ):
        temp = se.value_counts().to_dict()
        return max(temp, key=lambda x:temp[x])

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

    def plot_tree(self):
        pass



if __name__ == "__main__":
    df = pd.read_excel("TreeData.xlsx", index_col="id")
    id3_tree = ID3_tree(df)
    id3_tree = id3_tree.train(df)
    print(id3_tree.my_tree)


