#!/usr/bin/python3

import numpy as np
import pyglet
import pymunk
import math
from pymunk.pyglet_util import DrawOptions
from Pendulum import Linkage, Carriage, Floor, Target, PredictedLinkage


class Environment:
    def __init__(self,use = 'train'):
        self.use = use
        self.window_width = 1366
        self.window_height = 768

        # window = pyglet.window.Window(window_width,window_height)
        self.window = pyglet.window.Window(fullscreen=True)
        self.window_width = self.window.width
        self.window_height = self.window.height
        self.window.set_caption("RL Pendulum")

        self.fps_display = pyglet.window.FPSDisplay(window=self.window)
        self.batch = pyglet.graphics.Batch()

        self.options = DrawOptions()

        self.space = pymunk.Space()
        self.space.gravity = (0, -1000)

        self.floor = Floor(batch=self.batch, width=self.window_width, height=10,
                           x_pos=self.window_width//2, y_pos=100)
        self.floor.insert(self.space)

        self.carriage = Carriage(mass=1, batch=self.batch, x_pos=self.window_width//2,
                                 y_pos=self.floor.body.position[1])
        self.carriage.insert(self.space)

        self.linkage = Linkage(mass=0.1, height=250, aspect_ratio=0.09, x_pos=self.carriage.body.position[0],
                               y_pos=self.carriage.body.position[1]+125, batch=self.batch)
        self.linkage.insert(self.space)

        self.target = Target(x_pos=self.window_width//2, aspect_ratio=0.09,
                             y_pos=self.linkage.body.position[1], batch=self.batch)
        self.target.insert(self.space)

        self.joint = pymunk.constraint.PinJoint(
            self.linkage.body, self.carriage.body, anchor_a=(0, -self.linkage.height//2), anchor_b=(0, 0))
        self.joint.distance = 0
        self.space.add(self.joint)

        self.slider = pymunk.constraint.GrooveJoint(
            self.floor.body, self.carriage.body, groove_a=(-self.floor.width//2, 0), groove_b=(self.floor.width//2, 0), anchor_b=(0, 0))
        self.space.add(self.slider)

        if self.use == 'train':
            self.predicted_linkage = PredictedLinkage(x_pos=self.window_width//2, aspect_ratio=0.09,
                                                      y_pos=self.linkage.body.position[1], batch=self.batch)
            self.predicted_linkage.insert(self.space)
            self.train = False
            self.pure_model_prediction = False

        self.keyboard = pyglet.window.key.KeyStateHandler()
        self.window.push_handlers(self.keyboard)

        self.force = 0
        self.FORCE = 2000
        self.enable_random_force = False
        self.save_states = False

        @ self.window.event
        def on_draw():
            self.window.clear()
            self.space.debug_draw(self.options)
            self.fps_display.draw()
            self.batch.draw()

    def get_state(self):
        #angle = self.linkage.body.angle

        angle = math.fmod(self.linkage.body.angle, 2*math.pi)
        if(angle > math.pi):
            angle = -(2*math.pi-angle)
        elif(angle < -math.pi):
            angle = (2*math.pi + angle)

        state = np.array([self.carriage.body.position[0], angle,
                          self.carriage.body.velocity[0], self.linkage.body.angular_velocity])
        # print(self.linkage.body.angle, "\t", angle)

        return state

    def keyboard_handler(self):
        if self.enable_random_force is True:
            self.force = np.random.randn(1).item()*self.FORCE
        else:
            self.force = 0

        if(self.keyboard[pyglet.window.key.E] or self.keyboard[pyglet.window.key.RIGHT]):
            self.force += self.FORCE

        if(self.keyboard[pyglet.window.key.Q] or self.keyboard[pyglet.window.key.LEFT]):
            self.force += -self.FORCE

        if(self.keyboard[pyglet.window.key.SPACE]):
            self.target.random_position(x_range=[
                200, self.window_width-200], not_x_range=[self.window_width//2-300, self.window_width//2+300])

        if(self.keyboard[pyglet.window.key.R]):
            self.enable_random_force = False

        if(self.keyboard[pyglet.window.key.R] and self.keyboard[pyglet.window.key.LSHIFT]):
            self.enable_random_force = True
     
        if self.use == 'train':
            if(self.keyboard[pyglet.window.key.ENTER]):
                self.train = False

            if(self.keyboard[pyglet.window.key.ENTER] and self.keyboard[pyglet.window.key.LSHIFT]):
                self.train = True
             
            if(self.keyboard[pyglet.window.key.M]):
                self.pure_model_prediction = False

            if(self.keyboard[pyglet.window.key.M] and self.keyboard[pyglet.window.key.LSHIFT]):
                self.pure_model_prediction = True

            if(self.keyboard[pyglet.window.key.S]):
                self.save_states = True
