#!/usr/bin/python3

import pyglet
from Environment import Environment
from Policy import ValueIteration
import numpy as np
from utils import StatesEncoder, reward_function, get_all_states
from StateTransitionModel import StateTransitionModel
from tqdm import tqdm

env = Environment()
stm = StateTransitionModel(env.pendulums[-1]['physics'],dt = env.dt)

action_space = np.linspace(-300000, 300000, 5)
states_max = np.array([1920//2+300,0.5,500,10])
states_min = np.array([1920//2-300,-0.5,-500,-10])
divisions = 10
states_encoder = StatesEncoder(states_min = states_min, states_max = states_max, divisions=divisions)

policy = ValueIteration(actions=action_space,states_encoder=states_encoder,state_transition_model=stm,reward_function=reward_function,discount_factor = 0.99)

all_states = get_all_states(states_min, states_max, divisions*np.ones(4).astype(int))

ITERATION = 100
value_function = None

for iteration in range(ITERATION):
    print("Iteration:",iteration)
    policy.multiple_update_value_function(all_states)
    value_function = np.array(list(policy.value_function.values()))

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
