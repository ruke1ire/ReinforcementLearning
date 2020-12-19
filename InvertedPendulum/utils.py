#!/usr/bin/python3

import numpy as np

class StatesEncoder:
    def __init__(self,states_max, states_min, divisions):
        self.states_max = states_max
        self.states_min = states_min
        self.divisions = divisions

    def discretize_states(self,states):
        transformed_states = self.divisions*(states-self.states_min)/(self.states_max-self.states_min)
        transformed_states[transformed_states>self.divisions] = self.divisions-1
        transformed_states[transformed_states<0] = 1
        transformed_states = (transformed_states % self.divisions).astype(int)
        return transformed_states

    def get_key(self,states):
        discretized_states = self.discretize_states(states)
        return hash(discretized_states.tobytes())

def reward_function(states):
    angle = states[1]
    angular_velocity = states[3]
    x = states[0]
    reward = 10000/((angle)**2+1) + 100000/((960-x)**2+1) + 10000/((angular_velocity)**2+1)
    return reward

def get_all_states(states_min, states_max, divisions):
    series = []
    for mini, maxi,division in zip(states_min,states_max,divisions):
        series.append(np.linspace(mini,maxi,division))
    mesh = np.meshgrid(*series)
    final = []
    for m in mesh:
        final.append(m.reshape(-1))
    final = np.array(final).T
    return final

if __name__ == "__main__":
    states_max = np.array([1920,np.pi,1500,10])
    states_min = np.array([0,-np.pi,-1500,-10])
    divisions = 100

    se = StatesEncoder(states_max, states_min, divisions)

    states = np.array([2000,30,4,5])

    states_dict = {}
    states_dict[se.get_key(states)] = 5
    print(states_dict)
    print(states_dict[se.get_key(states)])
