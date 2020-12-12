#!/usr/bin/python3

import numpy as np
import pymunk
import pyglet
from pymunk.pyglet_util import DrawOptions

from Pendulum import Linkage, Carriage, Floor

# setup the window
window_width = 1366
window_height = 768
# window = pyglet.window.Window(window_width,window_height)
window = pyglet.window.Window(fullscreen=True)
window_width = window.width
window_height = window.height
window.set_caption("RL Pendulum")
fps_display = pyglet.window.FPSDisplay(window=window)

batch = pyglet.graphics.Batch()

# create drawoptions object
options = DrawOptions()

# setup space
space = pymunk.Space()
space.gravity = (0, -1000)

# create floor
floor = Floor(batch=batch, width=window_width, height=10,
              x_pos=window_width//2, y_pos=100)
floor.insert(space)

# create a carriage
carriage = Carriage(mass=1, batch=batch, x_pos=window_width//2,
                    y_pos=floor.body.position[1])
carriage.insert(space)

# create a linkage
linkage = Linkage(mass=0.1, height=250, aspect_ratio=0.09, x_pos=carriage.body.position[0],
                  y_pos=carriage.body.position[1]+125, batch=batch)
linkage.insert(space)


# Link the linkage and carriage
joint = pymunk.constraint.PinJoint(
    linkage.body, carriage.body, anchor_a=(0, -linkage.height//2), anchor_b=(0, 0))
joint.distance = 0
space.add(joint)

slider = pymunk.constraint.GrooveJoint(
    floor.body, carriage.body, groove_a=(-floor.width//2, 0), groove_b=(floor.width//2, 0), anchor_b=(0, 0))
space.add(slider)

keyboard = pyglet.window.key.KeyStateHandler()
window.push_handlers(keyboard)


@ window.event
def on_draw():
    window.clear()
    space.debug_draw(options)
    fps_display.draw()
    batch.draw()


def update(dt):
    FORCE = 2000
    force = 0

    if(keyboard[pyglet.window.key.E] or keyboard[pyglet.window.key.RIGHT]):
        force += FORCE
    if(keyboard[pyglet.window.key.Q] or keyboard[pyglet.window.key.LEFT]):
        force += -FORCE

    carriage.apply_force(force)

    linkage.update()
    carriage.update()

    space.step(1/60)


# Set pyglet update interval
pyglet.clock.schedule(update)
pyglet.app.run()
