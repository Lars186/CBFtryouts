# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:53:37 2023

@author: lesse
"""

import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

# constants
m = 2
k = 1
b = 1

x_0 = [0,0]
x_end = 5
t_end = 20

Vm = 2
Pm = 5
gamma = 20

def dynamics(x,u):
    # x = {position, velocity}
    dxdt = np.zeros_like(x)
    
    dxdt[0] = x[1]
    dxdt[1] = u/m
    
    return dxdt
def control(x):
    
    u = k*(x_end-x[0]) - b*x[1]
    
    return u
     
def simulate(dynamics,gamma, x_0, t_end, use_ASIF, dt):
    t = np.arange(0, t_end, dt)

    x_prev = np.array(x_0)
    u_prev = 0
    y = []
    u_out = []
    u_nom = []
    u_fil = []
    if use_ASIF:
        for ts in t:
            y.append(x_prev)
            # u = control_nom(x_prev)
            u_nominal = control(x_prev)
            u_filtered = asif(u_nominal,gamma,x_prev)
            if x_prev[1] > 0:
                x = dynamics(x_prev, u_filtered) * dt + x_prev
            else:
                x = dynamics(x_prev, u_nominal) * dt + x_prev
            x_prev = x
            u_nom.append(u_nominal)
            u_fil.append(u_filtered)
        y = np.array(y)
        u_nominal = np.array(u_nom)
        u_filtered = np.array(u_fil)
        
    else:
        for ts in t:
            y.append(x_prev)
            # u = control_nom(x_prev)
            u_nominal = control(x_prev)
            x = dynamics(x_prev, u_nominal) * dt + x_prev
            x_prev = x
            u_nom.append(u_nominal)
        y = np.array(y)
        u_nominal = np.array(u_nom)
        u_filtered = []
        
    return y, t, u_nominal, u_filtered

def _h(x):
    return Pm - x[0]

def _h_dot(x):
    return -x[0]

def cbf_cstr(gamma, x, u):
    return gamma * _h(x) + _h_dot(x)

def asif(u_nominal,gamma,x):
    # objective function, same for all CBF-QP
    p = np.array([1.0])
    q = np.array([-u_nominal])
    q = q.astype('float')
    g = np.array([1.0])
    g = g.astype('float')
    h = np.array([gamma*_h(x)])
    h = h.astype('float')
    
    u_filtered = solve_qp(p,q,g,h,solver = 'cvxopt')
    return u_filtered

y, t, u_nominal, u_filtered = simulate(dynamics,gamma, x_0, t_end,use_ASIF = 1,dt=0.1)


fig, axs = plt.subplots(2, 2)

axs[0,0].plot(t, y[:, 0])
axs[0,0].set_title('p')
axs[1,0].plot(t, y[:, 1])
axs[1,0].set_title('v')
axs[0,1].plot(y[:,0],y[:,1])
axs[0,1].set_title('p-v')
axs[1,1].plot(t,u_filtered)
plt.tight_layout()
plt.show()

y, t, u_nominal, u_filtered = simulate(dynamics,gamma, x_0, t_end,use_ASIF = 0,dt=0.1)


fig, axs = plt.subplots(2, 2)

axs[0,0].plot(t, y[:, 0])
axs[0,0].set_title('p')
axs[1,0].plot(t, y[:, 1])
axs[1,0].set_title('v')
axs[0,1].plot(y[:,0],y[:,1])
axs[0,1].set_title('p-v')
axs[1,1].plot(t,u_nominal)
axs[1,1].set_title('u [N]')
plt.tight_layout()
plt.show()

#fig, axs = plt.subplots(2, 2)
