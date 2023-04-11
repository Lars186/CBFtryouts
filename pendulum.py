# -*- coding: utf-8 -*-
"""
@author: lesse
"""
import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp, solve_ls

# constants

g = 9.81
m = 1
l = 0.5
I = m*l**2


t_end= 2
dt = 0.001

x0 = [0, 0]
x_end = np.pi

k = 100
b = 0
pmax = 3000.0

Em = 5
gamma = 75

def dynamics(x,u):
    # x = {position, velocity}
    dxdt = np.zeros_like(x)
    
    dxdt[0] = x[1]
    dxdt[1] = -g/l*np.sin(x[0])+(1/m*l**2)*u
    
    return dxdt

def control(x):
    
    u = k*(x_end-x[0]) - b*x[1]
    
    return u

def simulate(dynamics, gamma,x_0, t_end, dt, use_ASIF):
    t = np.arange(0, t_end, dt)

    x_prev = np.array(x_0)
    u_prev = 0
    y = []
    u = []
    Energy = []
    Epot = []
    Ekin = []
    unom = []
    hfunc = []
    pnom = []
    pcbf = []
    if use_ASIF:
        for ts in t:
            y.append(x_prev)
            u_nom = control(x_prev)
            u_filtered = asif(u_nom,gamma,x_prev,_h)
            x = dynamics(x_prev,u_filtered) * dt + x_prev
            x_prev = x
            u.append(u_filtered)
            Energy.append(E(x))
            Epot.append(Ep(x))
            Ekin.append(Ek(x))
            unom.append(u_nom)
            hfunc.append(_h(x))
            pnom.append(power(x,u_nom))
            pcbf.append(power(x,u_filtered))
        y = np.array(y)
        Epot = np.array(Epot)
        Ekin = np.array(Ekin)
        Energy = np.array(Energy)
        u_filtered = np.array(u)
        unom = np.array(unom)
        hfunc = np.array(hfunc)
        pnom = np.array(pnom)
        pcbf = np.array(pcbf)
    else:
        for ts in t:
            y.append(x_prev)
            u_nom = control(x_prev)
            x = dynamics(x_prev,u_nom) * dt + x_prev
            x_prev = x
            u.append(u_nom)
            Energy.append(E(x))
            Epot.append(Ep(x))
            Ekin.append(Ek(x))
            pnom.append(power(x,u_nom))
        y = np.array(y)
        Epot = np.array(Epot)
        Ekin = np.array(Ekin)
        Energy = np.array(Energy)
        u_filtered = []
        unom = np.array(u)
        pnom = np.array(pnom)
        
        
    return y, t, u_filtered, unom, Energy, Epot, Ekin, hfunc,pnom,pcbf

def _h(x):
    return Em - Ek(x)

def Ek(x):
    return 0.5*I*x[1]**2

def Ep(x): 
    return m*g*l*(1-np.cos(x[0]))

def E(x):
    return 0.5*I*x[1]**2 + m*g*l*(1-np.cos(x[0]))

def asif(u_nominal,gamma,x,_h):
    # objective function, same for all CBF-QP

    R = np.array([1.0])
    s = np.array([-u_nominal])
    s = s.astype('float')
    G = np.array([x[1]])
    G = G.astype('float')
    H = np.array([gamma*_h(x)])
    H = H.astype('float')
# =============================================================================
#     if  abs(x[1]) < 0.2:
#         lower = -1000
#         upper = 1000
#     elif x[1] < 0:
#         lower = -1*pmax/x[1]
#         upper = pmax/x[1]
#     elif x[1] > 0:
#         lower = -1*pmax/x[1]
#         upper = pmax/x[1]
#     else:
#         print("somethings is broken")
# =============================================================================
            
    #lower = np.array([-500]) #pmax/(abs(x[1]))
   #upper = np.array([75])
    lower = np.array([-75])
    upper = np.array([75])
    lower = lower.astype('float')
    upper = upper.astype('float')
    #u_filtered = solve_qp(p,q,g,h,lb,ub,solver = 'cvxopt')
    u_filtered = solve_qp(R,s,G,H,A=None,b=None,lb=lower,ub=upper,solver = 'cvxopt')
    return u_filtered

def power(x,u):
    return x[1]*u


ynom, t, u_filterednom, u_nom, E_nom, Epotnom, Ekin_nom, hval,pnom,pcbf = simulate(dynamics,gamma, x0, t_end, dt,use_ASIF = 0)

fig, axs = plt.subplots(2, 2)
#axs[1,1].plot(t, pnom, label = 'Nominal input power')
y, t, u_filtered, u_nomCBF, Energy, Epot, Ekin, hval,pnom,pcbf = simulate(dynamics,gamma, x0, t_end, dt,use_ASIF = 1)

#axs[0,0].plot(t, ynom[:,0], label = 'nominal control')
axs[0,0].plot(t, y[:,0], label = 'energy based CBF')
axs[0,0].set_xlabel('time [s]')
axs[0,0].set_title('Angle [rad]')
axs[0,0].legend()
#axs[0,1].plot(t, ynom[:,1], label = 'nominal control')
axs[0,1].plot(t, y[:,1], label = 'energy based CBF')
axs[0,1].set_xlabel('time [s]')
axs[0,1].set_title('velocity [rad/s]')
#axs[0,1].legend()
#axs[1,0].plot(t, u_nom, label = 'nominal control')
axs[1,0].plot(t, u_filtered, label = 'energy based CBF')
axs[1,0].set_xlabel('time [s]')
axs[1,0].set_title('torque input [nm]')
#axs[1,0].legend()
#axs[1,1].plot(t, pnom, label = 'Nominal input power')
axs[1,1].plot(t, pcbf, label = 'CBF input power')
axs[1,1].set_xlabel('time [s]')
axs[1,1].set_title('Input power ')
plt.tight_layout()
#axs[1,1].legend()

plt.savefig('Pendulum with CBFEm', dpi = 1500)
plt.show()

print("doing different gamma calculations...")
y, t, u_filtered, u_nomCBF, Energy, Epot, Ekin,hval,pnom,pcbf = simulate(dynamics,1, x0, t_end, dt,use_ASIF = 1)
plt.plot(t, hval, label = 'Gamma = 1'); print("Gamma = 1")
y, t, u_filtered, u_nomCBF, Energy, Epot, Ekin,hval,pnom,pcbf = simulate(dynamics,3, x0, t_end, dt,use_ASIF = 1)
plt.plot(t, hval, label = 'Gamma = 3'); print("Gamma = 3")
y, t, u_filtered, u_nomCBF, Energy, Epot, Ekin,hval,pnom,pcbf = simulate(dynamics,5, x0, t_end, dt,use_ASIF = 1)
plt.plot(t, hval, label = 'Gamma = 4'); print("Gamma = 5")
y, t, u_filtered, u_nomCBF, Energy, Epot, Ekin,hval,pnom,pcbf = simulate(dynamics,10, x0, t_end, dt,use_ASIF = 1)
plt.plot(t, hval, label = 'Gamma = 10'); print("Gamma = 10")
y, t, u_filtered, u_nomCBF, Energy, Epot, Ekin,hval,pnom,pcbf = simulate(dynamics,25, x0, t_end, dt,use_ASIF = 1)
plt.plot(t, hval, label = 'Gamma = 25'); print("Gamma = 25")
y, t, u_filtered, u_nomCBF, Energy, Epot, Ekin,hval,pnom,pcbf = simulate(dynamics,50, x0, t_end, dt,use_ASIF = 1)
plt.plot(t, hval, label = 'Gamma = 50'); print("Gamma = 50")
y, t, u_filtered, u_nomCBF, Energy, Epot, Ekin,hval,pnom,pcbf = simulate(dynamics,75, x0, t_end, dt,use_ASIF = 1)
plt.plot(t, hval, label = 'Gamma = 75'); print("Gamma = 75")
plt.xlabel('time [s]')
plt.ylabel('h(x) [J]')
plt.title('h(x) for different gamma, Em = 5')
plt.legend()
plt.savefig('Different Gammah', dpi = 1500)
plt.show()

# =============================================================================
# y, t, u_filtered, u_nomCBF, Energy, Epot, Ekin,hval,pnom,pcb = simulate(dynamics,1, x0, t_end, dt,use_ASIF = 1)
# plt.plot(t, y[:,0], label = 'Gamma = 1')
# y, t, u_filtered, u_nomCBF, Energy, Epot, Ekin,hval,pnom,pcb = simulate(dynamics,3, x0, t_end, dt,use_ASIF = 1)
# plt.plot(t, y[:,0], label = 'Gamma = 3')
# y, t, u_filtered, u_nomCBF, Energy, Epot, Ekin,hval,pnom,pcb = simulate(dynamics,5, x0, t_end, dt,use_ASIF = 1)
# plt.plot(t, y[:,0], label = 'Gamma = 5')
# y, t, u_filtered, u_nomCBF, Energy, Epot, Ekin,hval,pnom,pcb = simulate(dynamics,10, x0, t_end, dt,use_ASIF = 1)
# plt.plot(t, y[:,0], label = 'Gamma = 10')
# plt.xlabel('time [s]')
# plt.ylabel('angle [rad]')
# plt.title('Angle for different gamma, Em = 10')
# plt.legend()
# plt.savefig('Different Gamma angle', dpi = 1500)
# plt.show()
# =============================================================================

y, t, u_filtered, u_nomCBF, Energy, Epot, Ekin,hval,pnom,pcb = simulate(dynamics,1, x0, t_end, dt,use_ASIF = 1)
plt.plot(t, pcb, label = 'Gamma = 1')
y, t, u_filtered, u_nomCBF, Energy, Epot, Ekin,hval,pnom,pcb = simulate(dynamics,3, x0, t_end, dt,use_ASIF = 1)
plt.plot(t, pcb, label = 'Gamma = 3')
y, t, u_filtered, u_nomCBF, Energy, Epot, Ekin,hval,pnom,pcb = simulate(dynamics,5, x0, t_end, dt,use_ASIF = 1)
plt.plot(t, pcb, label = 'Gamma = 5')
y, t, u_filtered, u_nomCBF, Energy, Epot, Ekin,hval,pnom,pcb = simulate(dynamics,10, x0, t_end, dt,use_ASIF = 1)
plt.plot(t, pcb, label = 'Gamma = 10')
plt.xlabel('time [s]')
plt.ylabel('[power] [rad]')
plt.title('power for different gamma, Em = 10')
plt.legend()
plt.savefig('Different Gamma angle', dpi = 1500)
plt.show()

y, t, u_filtered, u_nomCBF, Energy, Epot, Ekin,hval,pnom,pcb = simulate(dynamics,75, x0, t_end, dt,use_ASIF = 1)

plt.plot(t,Ekin, label = 'kinetic energy with cbf')

y, t, u_filtered, u_nomCBF, Energy, Epot, Ekin,hval,pnom,pcbf = simulate(dynamics,75, x0, t_end, dt,use_ASIF = 0)
plt.plot(t,Ekin, label = 'kinetic energy without cbf')
plt.axhline(y=0.5, color='r', linestyle='-',label = "Barrier")
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('Energy [J]')
plt.title('Energy when CBF is applied')
plt.savefig('Pendulum with CBF, energy', dpi = 1200)
plt.show()


