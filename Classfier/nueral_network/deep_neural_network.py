#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy

import numpy as np

import matplotlib.pyplot as plt
from t3rdWeek.planar_utils import load_planar_dataset


def plot_decision_boundary(func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = func(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=plt.cm.Spectral)
    plt.show()


activ_func = {
    'identity': lambda x: x,
    'sigmoid': lambda x: 1./(1+np.exp(-x)),
    'relu': lambda x: (abs(x)+x)/2, # np.maximum(0, x),
    'tanh': lambda x: np.tanh(x),
}

d_activ_func = {
    'identity': lambda x: 1,
    'sigmoid': lambda x: activ_func['sigmoid'](x)*(1-activ_func['sigmoid'](x)),
    'relu': lambda x: (x>0).astype(float),
    'tanh': lambda x: 1 - np.power(np.tanh(x), 2),
}

class my_DNN(object):
    def __init__(self, hidden_layers_size=(3,), h_activation='relu',o_activation='sigmoid', output_layer_size=1, learning_rate=1e-2, max_iter=500, alpha=1e-4):
        """ my deep neural network
          In this module, the output activation function defaults to sigmoid.
        And the cost function defaults to logistic regression's cost function,
        so that only used to solve binary classification problem.

        parameters@
          hidden_layer_size: list-like, form layer 1 to layer L-1, ith element represent the number of ith layer's neurons unit;
          h_activation: Activation function for the hidden layer, default relu function
          learning_rate: learning rate 学习率
          max_iter: max iteration times
          alpha: penalty parameters 惩罚参数
        """
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.num_hidden_layers = len(hidden_layers_size) # L-1
        self.hidden_act_fun = activ_func[h_activation]
        self.hidden_dact_fun = d_activ_func[h_activation]
        self.output_act_fun = activ_func[o_activation]

        assert self.num_hidden_layers >= 0, 'have error hidden layer'
        self.depth = self.num_hidden_layers + 2 # L+1, at least 2 layers
        self.layers_size = [2,] + list(hidden_layers_size) + [output_layer_size,] \
                if not isinstance(hidden_layers_size, list) else hidden_layers_size  # all layers's neurons size

        self.parameters = {'W':[0, ] * self.depth, # input_layer_size default 3
                           'b':[0, ] * self.depth, }
        for ith in range(1, self.depth): # from layer 1 to L
            self.parameters['W'][ith] = np.random.randn(self.layers_size[ith], self.layers_size[ith-1]) * 0.01
            self.parameters['b'][ith] = np.zeros(shape=(self.layers_size[ith], 1))


    def train(self, X, Y, print_cost=False):
        self.X, self.Y = X, Y
        self.dim, self.m = X.shape
        self.parameters['W'][1] = np.random.randn(self.layers_size[1], self.dim) * 0.01

        for i in range(self.max_iter):
            A, Z = self.forw_propagation(X)
            cost = self.compute_cost(A[-1], Y)
            grads = self.back_propagation(A, Z)
            self.update_parameters(grads)

            if print_cost and i % 1000 == 0:
                print(i, "th iteration，the cost is：" + str(cost))


    def forw_propagation(self, X):
        W, b = self.parameters['W'], self.parameters['b']
        Z = [0, ]
        A = [ X ]

        for i in range(1, self.depth-1): # from layer 1 to L-1
            Z.append( np.dot(W[i], A[i-1]) + b[i] )
            A.append( self.hidden_act_fun(Z[i]) )
        # layer L, at this time: len(A)=L-1, len(W)=len(b)=L
        Z.append( np.dot(W[-1], A[-1]) + b[-1] )
        A.append( self.output_act_fun(Z[-1]) )
        return A, Z


    def compute_cost(self, AL, Y,):
        logprobs = np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL))
        cost = - np.sum(logprobs) / self.m
        cost = float(np.squeeze(cost))
        return cost


    def back_propagation(self, A, Z):
        W = self.parameters['W']
        m = self.m
        dZ = [0] * (self.num_hidden_layers + 2)
        dW = [0] * (self.num_hidden_layers + 2)
        db = [0] * (self.num_hidden_layers + 2)
        # layer L
        dZ[-1] =  A[-1] - self.Y
        dW[-1] =  (1/m) * np.dot(dZ[-1], A[-2].T)
        db[-1] =  (1/m) * np.sum(dZ[-1], axis=1, keepdims=True)

        for i in range(1, self.depth-1)[::-1]: # from layer L-1 to 1
            dZ[i] = np.multiply(np.dot(W[i+1].T, dZ[i+1]), self.hidden_dact_fun(Z[i]) )
            dW[i] = (1/m) * np.dot(dZ[i], A[i-1].T)
            db[i] = (1/m) * np.sum(dZ[i], axis=1, keepdims=True)

        grads = {"dW": dW, "db": db,}
        return grads


    def update_parameters(self, grads, ):
        dW, db = grads["dW"], grads["db"]
        for i in range(1, self.depth):
            self.parameters['W'][i] = self.parameters['W'][i] - self.learning_rate * dW[i]
            self.parameters['b'][i] = self.parameters['b'][i] - self.learning_rate * db[i]


    def predict(self, X):
        A, Z = self.forw_propagation(X, )
        predictions = np.round(A[-1])
        return predictions




if __name__ == '__main__':
    X, Y = load_planar_dataset()
    my_net = my_DNN(hidden_layers_size=(4, 3,), h_activation='tanh', max_iter=10000, learning_rate=0.5, )
    my_net.train(X, Y, print_cost=True)

    plot_decision_boundary(lambda x: my_net.predict(x), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))

    predictions = my_net.predict(X)
    # binary classification accuracy
    print ('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
