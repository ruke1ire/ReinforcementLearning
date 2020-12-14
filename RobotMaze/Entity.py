#!/usr/bin/python3

import numpy as np

class Entity:
    def __init__(self, environment, inititial_state = 0):
        self.environment = environment
        self.state = inititial_state
        self.reward = 0

    def move(self,action):
        next_state = self.environment.action_state(self.state,action)
        reward = self.environment.get_reward(next_state)
        self.reward += reward
        self.state = next_state

    def get_possible_actions(self):
        return ['up', 'down', 'left', 'right']

    def __repr__(self):
        environment = self.environment.environment.copy().reshape(-1)
        environment[self.state] = 2
        environment = environment.reshape(self.environment.environment.shape)

        s = ''
        for row in environment:
            for column in row:
                if column == 1:
                    s += ' . '
                elif column == 2:
                    s += ' o '
                else:
                    s += ' x '
            s += '\n'
        return s



        
