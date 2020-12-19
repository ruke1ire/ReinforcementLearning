#!/usr/bin/python3

import numpy as np
from tqdm import tqdm

class FittedValueIteration:
    def __init__(self,actions,state_transition_model):
        # array of possible actions
        self.actions = actions
        self.stm = state_transition_model

    def get_action(self):
        best_action = None
        best_value = -float('inf')
        for action in self.actions:
            state = self.stm.predict(action)
            value = self.get_value(state)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action
    
    def update_value_function(self):
        pass

    def get_value(self,state):
        return 0


class ValueIteration:
    def __init__(self,actions,states_encoder,state_transition_model,reward_function,discount_factor=0.9):
        self.actions = actions
        self.states_encoder = states_encoder
        self.stm = state_transition_model
        self.value_function = {}
        self.reward_function = reward_function
        self.discount_factor = discount_factor

    def get_action(self,current_states):
        best_action = None
        best_value = -float('inf')
        values = []
        for action in self.actions:
            states = self.stm.predict(current_states,action)
            value = self.get_value(states)
            values.append(value)
            if value > best_value:
                best_value = value
                best_action = action
        values = np.array(values)
        #best_action = np.mean(self.actions[values == np.max(values)])
#        print(best_action,'\t\t\t', best_value)
#        print(values)
        return best_action
    
    def update_value_function(self,current_states):
        #current_states = self.stm.pendulum.get_states()
        best_value = -float('inf')

        for action in self.actions:
            predicted_states = self.stm.predict(current_states,action)
            value = self.reward_function(predicted_states) + self.discount_factor*self.get_value(predicted_states)
            if value > best_value:
                best_value = value

        self.set_value(current_states, best_value)
    
    def multiple_update_value_function(self, multiple_current_states):
        best_values = []

        print("Finding best values")
        for current_states in tqdm(multiple_current_states):
            best_value = -float('inf')
            for action in self.actions:
                predicted_states = self.stm.predict(current_states,action)
                value = self.reward_function(predicted_states) + self.discount_factor*self.get_value(predicted_states)
                if value > best_value:
                    best_value = value
            best_values.append(best_value)

        for i,current_states in enumerate(multiple_current_states):
            self.set_value(current_states,best_values[i])


    def set_value(self,states,value):
        key = self.states_encoder.get_key(states)
        self.value_function[key] = value

    def get_value(self,states):
        key = self.states_encoder.get_key(states)
        if key in self.value_function:
            return self.value_function[key]
        else:
            return 0

class ValueIteration2:
    def __init__(self,actions,states_encoder,state_transition_model,reward_function,discount_factor=0.9):
        self.actions = actions
        self.states_encoder = states_encoder
        self.stm = state_transition_model
        self.value_function = []
        self.reward_function = reward_function
        self.discount_factor = discount_factor

    def get_action(self,current_states):
        best_action = None
        best_value = 0
        values = []
        for action in self.actions:
            states = self.stm.predict(current_states,action)
            value = self.get_value(states)
            values.append(value)
            if value > best_value:
                best_value = value
                best_action = action
        values = np.array(values)
        best_action = np.mean(self.actions[values == np.max(values)])
#        print(best_action,'\t\t\t', best_value)
#        print(values)
        return best_action
    
    def update_value_function(self,current_states):
        #current_states = self.stm.pendulum.get_states()
        best_value = 0

        for action in self.actions:
            predicted_states = self.stm.predict(current_states,action)
            value = self.reward_function(predicted_states) + self.discount_factor*self.get_value(predicted_states)
            if value > best_value:
                best_value = value

        self.set_value(current_states, best_value)
    
    def multiple_update_value_function(self, multiple_current_states):
        best_values = []

        print("Finding best values")
        for current_states in tqdm(multiple_current_states):
            best_value = 0
            for action in self.actions:
                predicted_states = self.stm.predict(current_states,action)
                value = self.reward_function(predicted_states) + self.discount_factor*self.get_value(predicted_states)
                if value > best_value:
                    best_value = value
            best_values.append(best_value)

        for i,current_states in enumerate(multiple_current_states):
            self.set_value(current_states,best_values[i])


    def set_value(self,states,value):
        key = self.states_encoder.get_key(states)
        self.value_function[key] = value

    def get_value(self,states):
        key = self.states_encoder.get_key(states)
        if key in self.value_function:
            return self.value_function[key]
        else:
            return 0
