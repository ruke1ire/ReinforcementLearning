#!/usr/bin/python3

import pyglet
from Environment import Environment
from Policy import ValueIteration
import numpy as np
from utils import StatesEncoder, reward_function
from StateTransitionModel import StateTransitionModel


env = Environment()
stm = StateTransitionModel(env.pendulums[-1]['physics'],dt = env.dt)

action_space = np.linspace(-3000000, 3000000, 21)
states_max = np.array([1920,np.pi,1500,10])
states_min = np.array([0,-np.pi,-1500,-10])
divisions = 20
states_encoder = StatesEncoder(states_min = states_min, states_max = states_max, divisions=divisions)

policy = ValueIteration(actions=action_space,states_encoder=states_encoder,state_transition_model=stm,reward_function=reward_function,discount_factor = 0.9)

def update(dt):
    force = policy.get_action()
    if force == None:
        force = 0
    env.pendulums[-1]['physics'].force = force

    env.update(dt)
    policy.update_value_function()
    #print(list(policy.value_function.values()))

pyglet.clock.schedule_interval(update,env.dt)
pyglet.app.run()
