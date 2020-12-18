#!/usr/bin/python3

from Pendulum import PendulumPhysics

class StateTransitionModel:
    def __init__(self, pendulum_physics,dt):
        self.pendulum = pendulum_physics
        self.simulated_pendulum = PendulumPhysics(self.pendulum.get_states())
        self.dt = dt

    def predict(self, force):
        self.simulated_pendulum.states = self.pendulum.get_states().copy()
        self.simulated_pendulum.force = force
        self.simulated_pendulum.step(self.dt)
        return self.simulated_pendulum.get_states()

