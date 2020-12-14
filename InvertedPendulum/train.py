#!/usr/bin/python3

import torch.optim as optim
import numpy as np
import pyglet
import math

from StateTransitionModel import StateTransitionModel
from Environment import Environment

env = Environment(use = 'train')

state_transition_model = StateTransitionModel()

state = [env.get_state()]
forces = [] 
prev_predicted_state = state[-1]

MAX_TRAINING_SIZE = 100000

def step():
    global state
    global forces
    s = np.array(state)
    prev_state = s[:-1]
    current_state = s[1:]

    f = np.array(forces)[:-1]
    if(env.train == True):
        state_transition_model.train_step(prev_state, f/env.FORCE, current_state)

def update(dt):
    global state
    global forces
    global prev_predicted_state


    training_size = 1

    env.keyboard_handler()

    forces.append(env.force)
    env.carriage.apply_force(env.force)

    env.space.step(1/60)

    env.linkage.update()
    env.carriage.update()


    if(len(state) > training_size) and (len(state) < MAX_TRAINING_SIZE):
        step()
    state.append(env.get_state())

    if(env.pure_model_prediction == False):
        predicted_state = state_transition_model.predict(
            state[-1], forces[-1]/env.FORCE).cpu().numpy()
    else:
        predicted_state = state_transition_model.predict(
            prev_predicted_state, forces[-1]/env.FORCE).cpu().numpy()

    prev_predicted_state = predicted_state

    env.predicted_linkage.move2(predicted_state)

pyglet.clock.schedule(update)
pyglet.app.run()
