#!/usr/bin/python3

import numpy as np

states_max = np.array([1920,np.pi,1500,10])
states_min = np.array([0,-np.pi,-1500,-10])
divisions = 100

states = np.array([2000,30,4,5])

def discretize_states(states):
    transformed_states = divisions*(states-states_min)/(states_max-states_min)
    transformed_states[transformed_states>divisions] = divisions-1
    transformed_states[transformed_states<0] = 1
    transformed_states = (transformed_states % divisions).astype(int)
    return transformed_states

transformed_states = discretize_states(states)
states_dict = {}
states_dict[hash(transformed_states.tobytes())] = 5
print(states_dict)
print(states_dict[hash(transformed_states.tobytes())])
