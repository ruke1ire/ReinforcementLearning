#!/usr/bin/python3

import pyglet
from Environment import Environment
import numpy as np

env = Environment()

def update(dt):
    env.update(dt)
    print("States:",env.pendulums[-1]['physics'].get_states())
    print()

pyglet.clock.schedule_interval(update,1/60)
pyglet.app.run()

