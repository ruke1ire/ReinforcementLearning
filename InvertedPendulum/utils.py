#!/usr/bin/python3

import hashlib
import numpy as np

class StatesEncoder:
    def __init__(self,states_max, states_min, divisions):
        self.states_max = states_max.astype(float)
        self.states_min = states_min.astype(float)
        self.divisions = divisions

    def discretize_states(self,states):
        states = states.astype(float)
        transformed_states = (self.divisions)*(states-self.states_min)/(self.states_max-self.states_min)
        transformed_states = (transformed_states).astype(int)
        transformed_states[transformed_states == self.divisions] = self.divisions-1
        return transformed_states

    def get_key(self,states):
        discretized_states = self.discretize_states(states)
        discretized_states.flags.writeable = False
        return discretized_states.data.tobytes()

def reward_function(states):
    angle = states[1]
    angular_velocity = states[3]
    x = states[0]
    vx = states[2]
    reward = 0
#    if (abs(angle) < 0.05):
#        reward = 1.0
#    if (abs(angle) > 0.1 or abs(vx) > 500):
#        reward = -1.0
    #reward = 10000000/((angle)**2+1) + 100000/((960-x)**2+1) + 10000/((angular_velocity)**2+1)
    reward = np.cos(angle) + np.cos(np.pi/2*(x-960)/960)
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
    states_max = np.array([4,10]).astype(float)
    states_min = np.array([0,-10]).astype(float)
    divisions = 2

    se = StatesEncoder(states_max, states_min, divisions)

    states = np.array([4,5])

    print(se.discretize_states(states))

    states_dict = {}
    states_dict[se.get_key(states)] = 5
    print(states_dict)
    print(states_dict[se.get_key(states)])




