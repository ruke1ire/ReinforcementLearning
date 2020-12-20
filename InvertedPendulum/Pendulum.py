#!/usr/bin/python3

import numpy as np
import pyglet
import math
import random

class PendulumPhysics:
    def __init__(self, initial_states = None, x_range = [0,1920]):
        if np.any(initial_states == None):
            initial_states = np.array([0,0,0,0])
        self.states = initial_states.astype(float)
        self.x_range = x_range
        self.max_velocity_x = 1500
        self.max_velocity_angle = 10
        self.clamp_position(self.states)
        self.clamp_velocity(self.states)
        self.clamp_angle(self.states)
        self.linkage_length = 240
        self.linkage_width = 10
        self.linkage_mass = 100
        self.carriage_mass = 100
        self.gravity = 1000
        self.force = 0
        self.reset_states()

    def clamp_velocity(self,states):
        if states[2] >= self.max_velocity_x:
            states[2] = self.max_velocity_x
        elif states[2] <= -self.max_velocity_x:
            states[2] = -self.max_velocity_x

        if states[3] >= self.max_velocity_angle:
            states[3] = self.max_velocity_angle
        elif states[3] <= -self.max_velocity_angle:
            states[3] = -self.max_velocity_angle

    def clamp_angle(self,states):
        angle = math.fmod(states[1], 2*math.pi)
        if(angle > math.pi):
            angle = -(2*math.pi-angle)
        elif(angle < -math.pi):
            angle = (2*math.pi + angle)

        states[1] = angle

    def clamp_position(self,states):
        if states[0] <= self.x_range[0]:
            states[0] = self.x_range[0]
            states[2] = 0.1*abs(states[2])
        elif states[0] >= self.x_range[1]:
            states[0] = self.x_range[1]
            states[2] = -0.1*abs(states[2])

    def step(self,dt):
        self.states = self.predict(self.states,self.force,dt)
        self.force = 0

    def predict(self, states, force, dt):
        A = np.array([
            [(self.linkage_mass+self.carriage_mass),-self.linkage_mass*self.linkage_length*np.cos(states[1])],
            [-np.cos(states[1]),self.linkage_length]])
        B = np.array([1,0]).reshape(-1,1)
        C = np.array([-self.linkage_mass*self.linkage_length*states[3]**2*np.sin(states[1])-17*states[2],self.gravity*np.sin(states[1])-20*states[3]]).reshape(-1,1)

        acc = np.linalg.inv(A)@(C+B*force)

        next_states = np.zeros_like(states)

        next_states[0] = states[0] + states[2]*dt + 1/2*acc[0]*(dt**2)
        next_states[1] = states[1] + states[3]*dt + 1/2*acc[1]*(dt**2)
        next_states[2] = states[2] + acc[0]*dt
        next_states[3] = states[3] + acc[1]*dt

        self.clamp_position(next_states)
        self.clamp_velocity(next_states)
        self.clamp_angle(next_states)
        return next_states

    def predict_multiple(self, multiple_states, multiple_forces, dt):
        next_states = []
        for states,force in zip(multiple_states, multiple_forces):
            next_states.append(self.predict(states,force,dt))
        next_states = np.array(next_states)
        return next_states

    def get_states(self):
        return self.states

    def reset_states(self):
        self.states = np.array([960,0,0,np.random.randn(1).item()/100]).astype(self.states.dtype)
        print('===============Reset States===============')

class PendulumVisualization:
    def __init__(self,pendulum_physics,batch):
        self.pendulum_physics = pendulum_physics
        self.batch = batch
        self.linkage_width = pendulum_physics.linkage_width
        self.linkage_length = pendulum_physics.linkage_length
        self.carriage_width = 40
        self.carriage_height = 20
        self.y = 300

        self._init_linkage()
        self._init_carriage()
    
    def _init_linkage(self):
        self.linkage = pyglet.shapes.Rectangle(0, 0, self.linkage_width, self.linkage_length, batch=self.batch)
        self.linkage.anchor_position = (self.linkage_width//2, 0)
        self.linkage.color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        self.linkage.opacity = 100

    def _init_carriage(self):
        self.carriage = pyglet.shapes.Rectangle(
            0, 0, self.carriage_width, self.carriage_height, batch=self.batch)
        self.carriage.anchor_position = (self.carriage_width//2, self.carriage_height//2)
        self.carriage.color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        self.carriage.opacity = 100

    def update(self):
        states = self.pendulum_physics.get_states()
        self.linkage.position = (states[0],self.y)
        self.linkage.rotation = -float(states[1])*180/3.1416

        self.carriage.position = (states[0],self.y)
        self.carriage.rotation = 0

if __name__ == "__main__":
    window = pyglet.window.Window(fullscreen=True)
    window_width = window.width
    window_height = window.height
    batch = pyglet.graphics.Batch()

    initial_state = np.array([window_width//2,0,0,0.1])

    pendulum_physics = PendulumPhysics(initial_states=initial_state)
    pendulum_visuals = PendulumVisualization(pendulum_physics,batch = batch)

    def update(dt):
        pendulum_physics.step(dt)
        pendulum_visuals.update()
        print("States:",pendulum_physics.get_states())

    @ window.event
    def on_draw():
        window.clear()
        batch.draw()

    pyglet.clock.schedule(update)
    pyglet.app.run()

