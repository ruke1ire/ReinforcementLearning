#!/usr/bin/python3

import pyglet
from Environment import Environment
from Policy import ValueIteration
import numpy as np
from utils import StatesEncoder, get_all_states
from StateTransitionModel import StateTransitionModel
import sys
import pickle

np.set_printoptions(threshold=sys.maxsize)

env = Environment()
stm = StateTransitionModel(env.pendulums[-1]['physics'],dt = env.dt)

action_space = np.linspace(-500000, 500000, 3)
states_max = np.array([1920,0.3,500,3])
states_min = np.array([0,-0.3,-500,-3])
divisions = 21
states_encoder = StatesEncoder(states_min = states_min, states_max = states_max, divisions=divisions)

policy = ValueIteration(actions=action_space,states_encoder=states_encoder,state_transition_model=stm)

policy.value_function = pickle.load(open(f"data/value_function-{divisions}.p","rb"))

def update(dt):

    force = policy.get_action(env.pendulums[-1]['physics'].states)
    if force == None:
        force = 0
    env.pendulums[-1]['physics'].force = force

    current_states = env.pendulums[-1]['physics'].get_states().copy()

    print(current_states)
    if abs(current_states[1] > 3):
        env.pendulums[-1]['physics'].reset_states()
        env.pendulums[-1]['physics'].force = 0

    env.update(dt)


pyglet.clock.schedule_interval(update,env.dt)
pyglet.app.run()


