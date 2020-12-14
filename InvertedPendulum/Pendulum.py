#!/usr/bin/python3

import pymunk
import random
import pyglet
import numpy as np
import math


class Linkage:
    def __init__(self, batch, mass=1, height=250, aspect_ratio=0.1, x_pos=100, y_pos=100):
        self.mass = mass
        self.aspect_ratio = aspect_ratio
        self.height = height
        self.width = int(self.height*self.aspect_ratio)
        self.moment = pymunk.moment_for_box(
            self.mass, (self.width, self.height))

        self.body = pymunk.Body(self.mass, self.moment)

        self.shape = pyglet.shapes.Rectangle(
            0, 0, self.width, self.height, batch=batch)
        self.shape.anchor_position = (self.width//2, self.height//2)
        self.shape.color = (random.randint(100, 255), random.randint(
            100, 255), random.randint(100, 255))
        self.shape.opacity = 100

        self.body.position = x_pos, y_pos
        self.body.angle = 0
        self.body.velocity = 0.0, 0.0
        self.body.angular_velocity = 0.0

    def update(self):
        self.body.angular_velocity = 0.99*self.body.angular_velocity

        self.shape.position = (self.body.position[0], self.body.position[1])
        self.shape.rotation = -float(self.body.angle)*180/3.1416

    def insert(self, space):
        space.add(self.body)

    def remove(self, space):
        space.remove(self.body)
        self.shape.delete()


class Carriage:
    def __init__(self, batch, mass, height=30, aspect_ratio=4, x_pos=100, y_pos=100):
        self.mass = mass
        self.height = height
        self.aspect_ratio = aspect_ratio
        self.width = int(self.height*self.aspect_ratio)
        self.moment = pymunk.moment_for_box(
            self.mass, (self.width, self.height))

        self.body = pymunk.Body(self.mass, self.moment)

        self.shape = pyglet.shapes.Rectangle(
            0, 0, self.width, self.height, batch=batch)
        self.shape.anchor_position = (self.width//2, self.height//2)
        self.shape.color = (random.randint(100, 255), random.randint(
            100, 255), random.randint(100, 255))
        self.shape.opacity = 100

        self.body.position = x_pos, y_pos
        self.body.angle = 0
        self.body.velocity = 0.0, 0.0

    def update(self):
        self.shape.position = (self.body.position[0], self.body.position[1])
        self.shape.rotation = -float(self.body.angle)*180/3.1416

    def insert(self, space):
        space.add(self.body)

    def remove(self, space):
        space.remove(self.body)
        self.shape.delete()

    def apply_force(self, force):
        self.body.apply_force_at_local_point(
            (force, 0), (0, 0))


class Floor:
    def __init__(self, batch, height=5, width=300, x_pos=100, y_pos=100):
        self.height = height
        self.width = width

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)

        self.shape = pyglet.shapes.Rectangle(
            0, 0, self.width, self.height, batch=batch)
        self.shape.anchor_position = (self.width//2, self.height//2)
        self.shape.color = (random.randint(100, 255), random.randint(
            100, 255), random.randint(100, 255))

        self.body.position = x_pos, y_pos
        self.body.angle = 0
        self.shape.position = (self.body.position[0], self.body.position[1])
        self.shape.rotation = -float(self.body.angle)*180/3.1416

    def insert(self, space):
        space.add(self.body)

    def remove(self, space):
        space.remove(self.body)
        self.shape.delete()


class Target:
    def __init__(self, batch, height=250, aspect_ratio=0.1, x_pos=100, y_pos=100):
        # set base body and shape
        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.height = height
        self.aspect_ratio = aspect_ratio
        self.width = int(self.height*self.aspect_ratio)

        self.shape = pyglet.shapes.Rectangle(
            0, 0, self.width, self.height, batch=batch)
        self.shape.anchor_position = (self.width//2, self.height//2)
        self.shape.color = (random.randint(100, 255), random.randint(
            100, 255), random.randint(100, 255))

        self.shape.opacity = 100

        # set base's position
        self.body.position = x_pos, y_pos

        self.index = 0

        self.update()

    def insert(self, space):
        space.add(self.body)

    def update(self):
        self.shape.position = (self.body.position[0], self.body.position[1])
        self.shape.rotation = -float(self.body.angle)*180/3.1416

    def iterate_position(self, reset=False, window_width=1000):
        x_positions = [int(window_width*0.1), int(window_width*0.9)]

        self.index = self.index % 2

        print(self.index)

        x_pos = x_positions[self.index]

        self.move(x_pos, self.body.position[1], 0)

        self.index += 1

        return self.index

    def random_position(self, x_range, not_x_range=[0, 0]):
        # randomly position the base
        # not x and not y should be a smaller window than x and y ranges
        if not ((x_range[0] < not_x_range[0]) or
                (x_range[1] > not_x_range[1])):
            raise ValueError("Invalid position ranges")

        x_pos = random.randint(x_range[0], x_range[1])

        while((x_pos > not_x_range[0]) and
                (x_pos < not_x_range[1])):
            x_pos = random.randint(x_range[0], x_range[1])

        self.move(x_pos, self.body.position[1], 0)

    def move(self, x, y, angle):
        self.body.position = x, y
        self.update()


class PredictedLinkage:
    def __init__(self, batch, height=250, aspect_ratio=0.1, x_pos=100, y_pos=100):
        # set base body and shape
        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.height = height
        self.aspect_ratio = aspect_ratio
        self.width = int(self.height*self.aspect_ratio)

        self.shape = pyglet.shapes.Rectangle(
            0, 0, self.width, self.height, batch=batch)
        self.shape.anchor_position = (self.width//2, self.height//2)
        self.shape.color = (random.randint(100, 255), random.randint(
            100, 255), random.randint(100, 255))

        self.shape.opacity = 100

        # set base's position
        self.body.position = x_pos, y_pos

        self.index = 0

        self.update()

    def insert(self, space):
        space.add(self.body)

    def update(self):
        self.shape.position = (self.body.position[0], self.body.position[1])
        self.shape.rotation = -float(self.body.angle)*180/3.1416

    def move(self, x, y, angle):
        self.body.position = x, y
        self.body.angle = angle
        self.update()

    def move2(self, state):
        angle = state[1]/100
        # if angle > 0:
        #    pass
        # else:
        #    angle = 2*math.pi + angle

        x_pos = state[0] - self.height/2*math.sin(angle)
        y_pos = 100+self.height/2*math.cos(angle)

        self.body.position = x_pos, y_pos
        self.body.angle = angle
        self.update()
