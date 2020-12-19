#!/usr/bin/python3

import pyglet
from Environment import Environment
from Policy import ValueIteration
import numpy as np
from utils import StatesEncoder, reward_function, get_all_states
from StateTransitionModel import StateTransitionModel
from tqdm import tqdm
import sys

np.set_printoptions(threshold=sys.maxsize)

env = Environment()
stm = StateTransitionModel(env.pendulums[-1]['physics'],dt = env.dt)

action_space = np.linspace(-250000, 250000, 3)
states_max = np.array([1920,0.3,1000,5])
states_min = np.array([0,-0.3,-1000,-5])
divisions = 10
states_encoder = StatesEncoder(states_min = states_min, states_max = states_max, divisions=divisions)

policy = ValueIteration(actions=action_space,states_encoder=states_encoder,state_transition_model=stm,reward_function=reward_function,discount_factor = 0.9)

all_states = get_all_states(states_min, states_max, divisions*np.ones(4).astype(int))

ITERATION = 30
value_function = None

for iteration in range(ITERATION):
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

def update(dt):

    force = policy.get_action(env.pendulums[-1]['physics'].states)
    if force == None:
        force = 0
    env.pendulums[-1]['physics'].force = force

    current_states = env.pendulums[-1]['physics'].get_states().copy()

    print(current_states)
    if np.any(current_states - states_min < 0) or np.any(states_max - current_states < 0):
        env.pendulums[-1]['physics'].reset_states()
        env.pendulums[-1]['physics'].force = 0

    env.update(dt)



pyglet.clock.schedule_interval(update,env.dt)
pyglet.app.run()

