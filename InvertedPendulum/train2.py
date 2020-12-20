#!/usr/bin/python3

import pyglet
from Pendulum import PendulumPhysics
from Environment import Environment
from Policy import ValueIteration
import numpy as np
from utils import StatesEncoder, reward_function, get_all_states
from StateTransitionModel import StateTransitionModel
from tqdm import tqdm
import sys
import pickle

np.set_printoptions(threshold=sys.maxsize)

pendulum_physics = PendulumPhysics()

stm = StateTransitionModel(pendulum_physics,dt = 1/60)

action_space = np.linspace(-500000, 500000, 3)
states_max = np.array([1920,0.3,500,3])
states_min = np.array([0,-0.3,-500,-3])
divisions = 21
states_encoder = StatesEncoder(states_min = states_min, states_max = states_max, divisions=divisions)

policy = ValueIteration(actions=action_space,states_encoder=states_encoder,state_transition_model=stm,reward_function=reward_function,discount_factor = 0.99)

all_states = get_all_states(states_min, states_max, divisions*np.ones(4).astype(int))

ITERATION = 10
try:
    policy.value_function = pickle.load(open(f"data/value_function-{divisions}.p","rb"))
except:
    pass

value_function = None

for iteration in range(1,ITERATION+1):
    print("Iteration:",iteration)
    policy.multiple_update_value_function(all_states)
    new_value_function = np.array(list(policy.value_function.values()))
    if value_function is not None:
        if np.all((new_value_function - value_function) < 1e-3):
            break
    value_function = new_value_function
    print(value_function)
    print(value_function.size)
    print(all_states.shape)
    print(np.sum(value_function != 0)/value_function.size)

pickle.dump(policy.value_function,open(f"data/value_function-{divisions}.p","wb"))
print("Training Completed")

