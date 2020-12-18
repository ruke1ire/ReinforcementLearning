#!/usr/bin/python3

import pyglet
from Environment import Environment
from Policy import ValueIteration
import numpy as np
from utils import StatesEncoder, reward_function
from StateTransitionModel import StateTransitionModel
from tqdm import tqdm


env = Environment()
stm = StateTransitionModel(env.pendulums[-1]['physics'],dt = env.dt)

action_space = np.linspace(-1000000, 1000000, 3)
states_max = np.array([1920,np.pi,1500,10])
states_min = np.array([0,-np.pi,-1500,-10])
divisions = 10
states_encoder = StatesEncoder(states_min = states_min, states_max = states_max, divisions=divisions)

policy = ValueIteration(actions=action_space,states_encoder=states_encoder,state_transition_model=stm,reward_function=reward_function,discount_factor = 0.9)

x_series = np.linspace(0,1920,divisions)
angle_series = np.linspace(-np.pi,np.pi,divisions)
xdot_series = np.linspace(-1500,1500,divisions)
anglular_velocity_series = np.linspace(-10,10,divisions)

xx, aa, xxdot, aadot = np.meshgrid(x_series, angle_series, xdot_series, anglular_velocity_series)
xx = xx.reshape(-1,1)
aa = aa.reshape(-1,1)
xxdot = xxdot.reshape(-1,1)
aadot = aadot.reshape(-1,1)

all_states = np.concatenate((xx,aa,xxdot,aadot),axis=1)

ITERATION = 20

for iteration in range(ITERATION):
    print(iteration)
    for states in tqdm(all_states):
        policy.update_value_function(states)
    print(list(policy.value_function.values()))
    



def update(dt):
    force = policy.get_action()
    if force == None:
        force = 0
    env.pendulums[-1]['physics'].force = force
    #print('force',force)

    env.update(dt)
    #policy.update_value_function()
    #print(list(policy.value_function.values()))

pyglet.clock.schedule_interval(update,env.dt)
pyglet.app.run()
