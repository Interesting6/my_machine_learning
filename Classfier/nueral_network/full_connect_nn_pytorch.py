#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy

import os, glob, re
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing as skp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import LeaveOneOut, KFold, train_test_split

import torch
from torch import nn, optim
from torch.nn import functional as F



class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, net_size=(10,)):
        super(DNN, self).__init__()
        self.net_length = len(net_size)
        self.net_size = net_size
        self.fc0 = nn.Linear(input_dim, net_size[0])
        self.fc = nn.ModuleList([self.fc0, ])
        for i in range(self.net_length - 1):
            self.fc.append(nn.Linear(net_size[i], net_size[i + 1]))
        self.fc.append(nn.Linear(net_size[-1], output_dim))

    def forward(self, x):
        for i in range(self.net_length + 1):
            x = F.relu(self.fc[i](x))
        return x


if __name__ == '__main__':
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    data_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # 分别为单通道的均值与标准差
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf,) # download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=data_tf)
    batch_size = 64
    learning_rate = 1e-2
    num_epoches = 20

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    fcn_model = DNN(28 * 28, 10, (300, 100))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fcn_model = fcn_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(fcn_model.parameters(), lr=learning_rate)

    epoch = 0
    for data in train_loader:
        img, label = data
        batch_size = img.size(0)
        img = img.view(batch_size, -1)
        img = img.to(device); label = label.to(device)
        logits = fcn_model(img)
        loss = criterion(logits, label)
        print_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch += 1
        if epoch % 50 == 0:
            print('epoch: {}, loss: {:.4}'.format(epoch, print_loss))

    print('training process has been done!')

    fcn_model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        batch_size = img.size(0)
        img = img.view(batch_size, -1)
        img = img.to(device); label = label.to(device)
        logits = fcn_model(img)
        loss = criterion(logits, label)
        eval_loss += loss.item()*batch_size
        _, pred = torch.max(logits, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()

    eval_loss = eval_loss / len(test_dataset)
    eval_acc = eval_acc / len(test_dataset)

    print('loss: {:.4f}, acc: {:.4f}'.format(eval_loss, eval_acc))




