#!/usr/bin/python3

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

    def get_action(self):
        best_action = None
        best_value = -100000
        for action in self.actions:
            states = self.stm.predict(action)
            value = self.get_value(states)
            if value > best_value:
                best_value = value
                best_action = action
        print(best_value)
        return best_action
    
    def update_value_function(self):
        current_states = self.stm.pendulum.get_states()
        best_value = -100000

        for action in self.actions:
            predicted_states = self.stm.predict(action)
            value = self.reward_function(current_states) + self.discount_factor*self.get_value(predicted_states)
            if value > best_value:
                best_value = value

        self.set_value(current_states, best_value)

    def set_value(self,states,value):
        key = self.states_encoder.get_key(states)
        self.value_function[key] = value

    def get_value(self,states):
        key = self.states_encoder.get_key(states)
        if key in self.value_function:
            return self.value_function[key]
        else:
            return -float('inf')