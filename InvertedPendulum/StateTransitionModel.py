#!/usr/bin/python3

import torch
import torch.nn as nn


class StateTransitionModel(nn.Module):
    def __init__(self):
        super().__init__()

