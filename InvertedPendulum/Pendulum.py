#!/usr/bin/python3

import pymunk
import random
import pyglet


class Linkage:
    def __init__(self, batch, mass=1, height=200, aspect_ratio=0.1, x_pos=100, y_pos=100):
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
        #self.shape.opacity = 100

        self.body.position = x_pos, y_pos
        self.body.angle = 0
        self.body.velocity = 0.0, 0.0
        self.body.angular_velocity = 0.1

    def update(self, text=None):
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

    def update(self, text=None):
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
