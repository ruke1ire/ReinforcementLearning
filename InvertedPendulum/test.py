#!/usr/bin/python3

import numpy as np

divisions = 10

x_series = np.linspace(0,1920,divisions)
angle_series = np.linspace(-np.pi,np.pi,divisions)
xdot_series = np.linspace(-1500,1500,divisions)
anglular_velocity_series = np.linspace(-10,10,divisions)

xx, aa, xxdot, aadot = np.meshgrid(x_series, angle_series, xdot_series, anglular_velocity_series)
xx = xx.reshape(-1,1)
aa = aa.reshape(-1,1)
xxdot = xxdot.reshape(-1,1)
aadot = aadot.reshape(-1,1)

all_states = np.concatenate((xx,aa,xxdot,aadot),axis=1)
print(all_states.shape)
