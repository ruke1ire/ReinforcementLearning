#!/usr/bin/python3

# the transition model models the non-linear dynamics of the inverted pendulum
# it should have an input of all the states as well as the inputs and it should try to predict
# the state at the next iteration
# it trains after every action taken by the policy

import torch
import torch.nn as nn


class StateTransitionModel(nn.Module):
    def __init__(self):
        super().__init__()

