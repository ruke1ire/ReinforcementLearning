#!/usr/bin/python3

import numpy as np
import pyglet

from Environment import Environment

env = Environment(use='manual')

def update(dt):
    state = env.get_state()
    env.keyboard_handler()

    env.carriage.apply_force(env.force)
    env.space.step(1/60)
    env.linkage.update()
    env.carriage.update()


# Set pyglet update interval
pyglet.clock.schedule(update)
pyglet.app.run()
