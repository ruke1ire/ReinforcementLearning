#!/usr/bin/python3

# the transition model models the non-linear dynamics of the inverted pendulum
# it should have an input of all the states as well as the inputs and it should try to predict
# the state at the next iteration
# it trains after every action taken by the policy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class StateTransitionModel:
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_1 = nn.Sequential(nn.Linear(5, 4))

        def forward(self, x):
            x = self.fc_1(x)
            return x

    def __init__(self):
        self.net = self.Net().cuda()
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=0.01)

    def train_step(self, prev_state, action, current_state):
        TILE = 1
        self.net.train()

        X = np.append(prev_state, action[:,np.newaxis],axis=1)
        X = np.tile(X, (TILE, 1))
        X = torch.Tensor(X).cuda()

        y = current_state[:,:]
        y = np.tile(y, (TILE, 1))
        noise = np.random.randn(y.shape[0],y.shape[1])/10000
        y = y+noise
        y = torch.Tensor(y).cuda()
        print(y.shape)

        self.net.zero_grad()
        output = self.net(X)
        loss = self.loss_function(output, y)

        loss.backward()
        self.optimizer.step()

        self.net.eval()

    def predict(self, prev_state, action):
        X = np.append(prev_state, action)
        X = torch.Tensor(X).view(-1).cuda()

        with torch.no_grad():
            output = self.net(X)
        return output

