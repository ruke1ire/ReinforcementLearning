#!/usr/bin/python3

import pyglet
from Pendulum import PendulumPhysics, PendulumVisualization
import numpy as np

class Environment:
    def __init__(self):
        self.dt = 1/60
        self._init_window()
        self._init_pendulum()
         
        @ self.window.event
        def on_draw():
            self.window.clear()
            self.batch.draw()
            self.fps_display.draw()

    def _init_window(self):
        self.window = pyglet.window.Window(fullscreen=True)
        self.window_width = self.window.width
        self.window_height = self.window.height
        self.batch = pyglet.graphics.Batch()
        self.keyboard = pyglet.window.key.KeyStateHandler()
        self.window.push_handlers(self.keyboard)
        self.fps_display = pyglet.window.FPSDisplay(window=self.window)

    def _init_pendulum(self):
        self.pendulums = []
        self.insert_pendulum()

    def insert_pendulum(self, initial_states = None):
        pendulum_physics = PendulumPhysics(initial_states=np.array([self.window_width//2,0,0,0]))
        pendulum_visuals = PendulumVisualization(pendulum_physics,batch = self.batch)
        pendulum = {'physics':pendulum_physics,'visuals':pendulum_visuals}
        self.pendulums.append(pendulum)

    def keyboard_handler(self):
        force = self.pendulums[0]['physics'].force
        if(self.keyboard[pyglet.window.key.E] or self.keyboard[pyglet.window.key.RIGHT]):
            force += 300000

        if(self.keyboard[pyglet.window.key.Q] or self.keyboard[pyglet.window.key.LEFT]):
            force -= 300000
        
        self.pendulums[0]['physics'].force = force

    def update(self,dt):
        self.keyboard_handler()
        for pendulum in self.pendulums:
            pendulum['physics'].step(dt)
            if pendulum['visuals']:
                pendulum['visuals'].update()

