#!/usr/bin/python3

import numpy as np
import time
import os

from BoardGame import Environment, Entity, Policies

board = np.array([
    [1, 0, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0],
    [1, 0, 1, 1, 1, 0],
    [1, 1, 1, 0, 1, 1]])

reward = np.array([
    [10, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]])

env = Environment.Environment(board, reward)
entity = Entity.Entity(env, inititial_state=board.size-1)
policy = Policies.ValueIterationPolicy(entity, discount_factor=0.999)
prev_state = entity.state

while True:
    os.system('clear')
    print()
    print(entity)
    action = policy.get_action(entity.state)
    entity.move(action)
    if prev_state == entity.state:
        break
    prev_state = entity.state
    time.sleep(0.1)
