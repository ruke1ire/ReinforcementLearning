#!/usr/bin/python3

import numpy as np
from numpy import ndarray

class Environment:
    def __init__(self, environment: ndarray, reward: ndarray):
        #takes a numpy arrays of 0s and 1s as an input
        assert np.max(environment) <= 1
        assert np.min(environment) >= 0

        self.environment = environment
        self.environment_vector = environment.reshape(-1)
        self.environment_shape = self.environment.shape

        #reward is a numpy array with the rewards for being in a state
        assert environment.shape == reward.shape

        self.reward = reward
        self.reward_vector = reward.reshape(-1)

    def __repr__(self):
        s = ''
        for row in self.environment:
            for column in row:
                if column == 1:
                    s += ' . '
                else:
                    s += ' x '
            s += '\n'
        return s

    def action_state_calculation(self,current_state, action):
        # arithmetic calculation for the next state
        if (action == 'up'):
            temp_state = current_state - self.environment_shape[1]
        elif (action == 'down'):
            temp_state = current_state + self.environment_shape[1]
        elif (action == 'left'):
            temp_state = current_state - 1
        elif (action == 'right'):
            temp_state = current_state + 1
        return temp_state

    def action_state(self, current_state: int, action: str):
        #returns the next possible state after an action was taken
        #action 'up' 'down' 'right' 'left'
        possible_states = self._valid_states(current_state)

        if (action == 'up') and ('up' in possible_states):
            temp_state = self.action_state_calculation(current_state, 'up')
        elif (action == 'down') and ('down' in possible_states):
            temp_state = self.action_state_calculation(current_state, 'down')
        elif (action == 'left') and ('left' in possible_states):
            temp_state = self.action_state_calculation(current_state, 'left')
        elif (action == 'right') and ('right' in possible_states):
            temp_state = self.action_state_calculation(current_state, 'right')
        else:
            temp_state = current_state

        next_state = temp_state

        return next_state

    def get_reward(self, current_state: int):
        if self._valid_state(current_state):
            return self.reward_vector[current_state]
        else:
            raise ValueError('invalid state')

    def _valid_states(self,current_state):
        # checks to see all the possible states to move from the current state
        possible_actions = ['up','down','left','right']

        # in the first row
        if(current_state < self.environment_shape[1]):
            if 'up' in possible_actions:
                possible_actions.remove('up')
        elif(self.environment_vector[self.action_state_calculation(current_state,'up')] == 0):
            if 'up' in possible_actions:
                possible_actions.remove('up')

        # in the last row
        if(current_state >= self.environment_vector.size - self.environment_shape[1]):
            if 'down' in possible_actions:
                possible_actions.remove('down')
        elif(self.environment_vector[self.action_state_calculation(current_state, 'down')] == 0):
            if 'down' in possible_actions:
                possible_actions.remove('down')

        # in the first column
        if(current_state % self.environment_shape[1] == 0):
            if 'left' in possible_actions:
                possible_actions.remove('left')
        elif(self.environment_vector[self.action_state_calculation(current_state,'left')] == 0):
            if 'left' in possible_actions:
                possible_actions.remove('left')

        # in the last column
        if(current_state % self.environment_shape[1] == self.environment_shape[1] - 1):
            if 'right' in possible_actions:
                possible_actions.remove('right')
        elif(self.environment_vector[self.action_state_calculation(current_state,'right')] == 0):
            if 'right' in possible_actions:
                possible_actions.remove('right')

        return possible_actions

    def _valid_state(self,state) -> bool:
        # checks to see if a state is valid
        if (state >= self.environment_vector.size) or (state < 0):
            return False
        else:
            return True

    def get_possible_states(self):
        return np.arange(self.environment_vector.size)



        




