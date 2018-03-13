import numpy as np
import pandas as pd

class my_naive_bayes_smooth(object):
    """docstring for my_naive_bayes"""
    def __init__(self, df):
        super(my_naive_bayes_smooth, self).__init__()
        self.df = df
        self.X_train = df.iloc[:,:-1]
        self.y_train = df.iloc[:,-1]
        self.label_set = set(self.y_train)
        self.features = df.columns[:-1]
        self.label_name = df.columns[-1]
        self.feature_dict = {}
        self.n_sample = len(df)

    def get_prior_p(self, g):
        prior_p = {}
        for label in self.label_set:
            counts = g.size() + 1 # 平滑
            prior_p[label] = counts[label] / sum(counts)  
        return prior_p

    def get_cond_p(self, g):
        cond_p = {}
        for label, group in g:
            cond_p[label] = {}
            for feature in self.features:
                counts = group[feature].value_counts() + 1 # 平滑
                cond_p[label][feature] = counts / sum(counts)
        return cond_p

    def train(self, ):
        for feature in self.features:
            self.feature_dict[feature] = set(self.df[feature])
        g = self.df.groupby(self.label_name)

        self.prior_p = self.get_prior_p(g)
        self.cond_p = self.get_cond_p(g)
        return self

    def predict_one(self, test_X):
        semi_post_p = {}
        for label in self.label_set:
            temp = 1
            for feature in self.features:
                temp = temp * self.cond_p[label][feature][test_X[feature]]
            semi_post_p[label] = self.prior_p[label] * temp
        return max(semi_post_p, key=semi_post_p.get)


if __name__ == '__main__':
    df = pd.read_excel("bayes_data.xlsx",index_col="index")

    bayes2 = my_naive_bayes_smooth(df)
    bayes2 = bayes2.train()

    test_x = df.loc[6]
    label = bayes2.predict_one(test_x)
    print(label)

