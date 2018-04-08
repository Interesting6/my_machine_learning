#!/usr/bin/env python 
# -*- coding: utf-8 -*-
""" 
@version: py3.5        @license: Apache Licence  
@author: 'Treamy'    @contact: chenymcan@gmail.com 
@file: CTree.py      @software: PyCharm 
@time: 2018/3/23 22:12 @site: www.ymchen.cn
"""

import numpy as np
import pandas as pd


class ClassifyTree(object):
    def __init__(self, feat_index=-1, feat=None, results=None, right=None, left=None):
        self.feat_index = feat_index
        self.feat = feat
        self.results = results
        self.right = right
        self.left = left

    def calc_gini(self,df, ):
        # 对category分类计算各个样本所占比
        P_ = df[self.cate].value_counts() / df[self.cate].count()
        return 1-sum(map(lambda p: p**2, P_))

    def calc_cond_gini(self, feat, feat_value):
        n = self.df[feat].count()
        is_value_index = self.df[feat] == feat_value
        P_feat_value = sum(is_value_index) / n
        is_value_df = self.df[is_value_index]
        is_value_gini = self.calc_gini(is_value_df, )

        not_value_index = self.df[feat] != feat_value
        P_not_feat_value = sum(not_value_index) / n
        not_value_df = self.df[not_value_index]
        not_value_gini = self.calc_gini(not_value_df)

        gini = P_feat_value*is_value_gini + P_not_feat_value*not_value_gini
        return gini

    def build_tree(self, df):
        feats, cate = df.columns[:-1].tolist(), df.columns[-1]
        dic = {}
        for feat in feats:
            dic[feat] = {}
            feat_values = df[feat].unique()
            for feat_value in feat_values:
                dic[feat][feat_value] = self.calc_cond_gini(feat, feat_value)
                
        return


    def train(self, df):
        self.df = df.copy()
        self.feats, self.cate = df.columns[:-1], df.columns[-1]
        feats, cate = df.columns[:-1].tolist(), df.columns[-1]
        Gini_D = self.calc_gini(df)







if __name__ == "__main__":
    tree = ClassifyTree()


