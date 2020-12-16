#!/usr/bin/python3

# the transition model models the non-linear dynamics of the inverted pendulum
# it should have an input of all the states as well as the inputs and it should try to predict
# the state at the next iteration
# it trains after every action taken by the policy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class StateTransitionModel:
    class Net(nn.Module):
        def __init__(self):
            self.INPUT_FEATURES = 6
            self.OUTPUT_FEATURES = 5
            super().__init__()
            self.fc_1 = nn.Sequential(nn.Linear(self.INPUT_FEATURES, self.OUTPUT_FEATURES))

        def forward(self, x):
            x = self.fc_1(x)
            return x

    def __init__(self):
        self.net = self.Net().cuda()
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=0.01)
        self.fig,self.ax = plt.subplots(1,1)
        self.step = 0


    def plot_animate(self,lines,labels = None):
        if labels == None:
            labels = ['None']*lines.shape[1]
        mean = np.mean(lines,axis=0)
        lines = (lines-mean)
        self.ax.clear()
        series = np.arange(lines.shape[0])
        for i in range(lines.shape[1]):
            self.ax.plot(series, lines[:,i],label = labels[i])
        self.ax.legend()
        plt.pause(0.00001)

    def train_step(self, prev_state, action, current_state):
        self.step += 1
        TILE = 1
        self.net.train()

        sin_angle = np.sin(prev_state[:,1])
        cos_angle = np.cos(prev_state[:,1])
        
        X = np.concatenate((prev_state[:,0][:,np.newaxis], sin_angle[:,np.newaxis],cos_angle[:,np.newaxis],prev_state[:,[2,3]]),axis=1)
        X = np.append(X,action[:,np.newaxis],axis=1)

#        X = np.append(prev_state, action[:,np.newaxis],axis=1)
        X = np.tile(X, (TILE, 1))

        current_sin_angle = np.sin(current_state[:,1])
        current_cos_angle = np.cos(current_state[:,1])

        y = np.concatenate((current_state[:,0][:,np.newaxis], current_sin_angle[:,np.newaxis],current_cos_angle[:,np.newaxis],current_state[:,[2,3]]),axis=1)

        y = np.tile(y, (TILE, 1))
        #noise = np.random.randn(y.shape[0],y.shape[1])/100
        #y = y+noise

        #print(X[-1,:-1],"\t\t\t",y[-1,:])

#        if(self.step % 100 == 0):
#            labels = ['X','sin','cos','Vx','Angular Velocity']
#            self.plot_animate(X[-300:,:-1],labels)

        y = torch.Tensor(y).cuda()
        X = torch.Tensor(X).cuda()

        self.net.zero_grad()
        output = self.net(X)
        loss = self.loss_function(output, y)

        loss.backward()
        self.optimizer.step()

        self.net.eval()

    def predict(self, prev_state, action):
        prev_state = prev_state[np.newaxis,:]
        action = np.array(action).reshape(-1)

        sin_angle = np.sin(prev_state[:,1])
        cos_angle = np.cos(prev_state[:,1])
        
        X = np.concatenate((prev_state[:,0][:,np.newaxis], sin_angle[:,np.newaxis],cos_angle[:,np.newaxis],prev_state[:,[2,3]]),axis=1)
        X = np.append(X,action[:,np.newaxis],axis=1)

        with torch.no_grad():
            X = torch.Tensor(X).cuda()
            output = self.net(X)

        output = output.cpu().numpy()
        angle = np.arctan2(output[:,1],output[:,2])
        output = np.concatenate((output[:,0][:,np.newaxis],angle[:,np.newaxis],output[:,3:]),axis=1)

        if(output.size == output.shape[1]):
            return output[0]
        else:
            return output

