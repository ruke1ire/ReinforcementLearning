#!/usr/bin/python3

from Pendulum import PendulumPhysics

class StateTransitionModel:
    def __init__(self, pendulum_physics,dt):
        self.pendulum = pendulum_physics
        self.simulated_pendulum = PendulumPhysics(self.pendulum.get_states())
        self.dt = dt

    def predict(self, current_states,force):
        self.simulated_pendulum.states = current_states
        self.simulated_pendulum.force = force
        self.simulated_pendulum.step(self.dt)
        return self.simulated_pendulum.get_states()
    
    def predict_multiple(self, multiple_current_states, multiple_forces):
        return self.simulated_pendulum.predict_multiple(multiple_current_states,multiple_forces,self.dt)


