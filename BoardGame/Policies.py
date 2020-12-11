#!/usr/bin/python3

import numpy as np
from numpy import ndarray

class Policy:
    def __init__(self, entity):
        self.entity = entity
        self.environment = self.entity.environment
        self.possible_actions = self.entity.get_possible_actions()
        self.possible_states = self.environment.get_possible_states()

    def get_action(self, current_state: int) -> str:
        raise NotImplementedError()

class ValueIterationPolicy(Policy):
    def __init__(self, entity, discount_factor = 0.9):
        super().__init__(entity)

        self.discount_factor = discount_factor
        self.value = np.zeros_like(self.possible_states)
        self.value_iteration()

    def value_iteration(self):
        #value iteration iteration (synchronous) algorithm here
        #TODO: make get_reward return an array, and make finding the value of an a matrix (parallel)
        self.value = np.zeros_like(self.possible_states)
        while True:
            R = []
            V_max = []

            for state in self.possible_states:
                R.append(self.environment.get_reward(state))
                V_action = []
                for action in self.possible_actions:
                    V_action.append(self.value[self.environment.action_state(state, action)])

                V_action = np.array(V_action)
                V_max.append(np.max(V_action))

                if (self.environment.environment_vector[state] == 0):
                    V_max[-1] = 0
#                if (self.environment.reward_vector[state] != 0):
#                    V_max[-1] = 0

            R = np.array(R)
            V_max = np.array(V_max)
            self.prev_value = self.value
            self.value = R + self.discount_factor*V_max
            if np.sum(np.abs(self.prev_value - self.value)) < 1e-3:
                break
        print()
        print(self.value.reshape(self.environment.environment_shape))

    def get_action(self,current_state):
        V_action = []
        for action in self.possible_actions:
            V_action.append(self.value[self.environment.action_state(current_state, action)])

        V_action = np.array(V_action)
        arg_action = np.argmax(V_action)

        return self.possible_actions[arg_action]

class PolicyIterationPolicy(Policy):
    def __init__(self, possible_actions):
        super().__init__(possible_actions)

    def policy_iteration(self, possible_states):
        #policy iteration algorithm here
        self.possible_states = possible_states
        value = np.zeros_like(self.possible_states)
        pass

    def get_action(self,current_state):
        pass

class RandomPolicy(Policy):
    def __init__(self, possible_actions):
        super().__init__(possible_actions)

    def get_action(self,current_state):
        import random
        action = random.choice(self.possible_actions)
        return action

