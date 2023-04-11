# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:15:45 2023

@author: lesse
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 13:53:06 2023

@author: lesse
"""
import sympy as sp
from sympy import sin, cos, simplify
from sympy import symbols as syms
from sympy.matrices import Matrix
from sympy.utilities.lambdify import lambdify
import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

# state, state derivative, and control variables
x_1, x_2, xdot_1, xdot_2, xddot_1, xddot_2, f = syms('x_1, x_2, xdot_1, xdot_2, xddot_1, xddot_2, f ')

l_1, m_c, m_1, g = 1.0,5.0,1.0,9.81
Emax = 2
gamma = 0.1
use_asif = 1

# Create Generalized Coordinates as a function of time: x = [theta_1, theta_2, theta_3]
p = Matrix([l_1, m_c, m_1, g])      # parameter vector

x = Matrix([x_1, x_2])                   # generalized positions
xdot = Matrix([xdot_1, xdot_2])       # time derivative of x
xddot = Matrix([xddot_1, xddot_2])   # time derivative of xdot  

# kinematics:
p_c = Matrix([x_1, 0])
p_1 = p_c + l_1/2 * Matrix([cos(x_2), sin(x_2)])

v_c = p_c.jacobian(Matrix([x_1])) * Matrix([xdot_1])
v_1 = p_1.jacobian(Matrix([x_1, x_2])) * Matrix([xdot_1, xdot_2])

K_c = m_c * v_c.T*v_c / 2
K_1 = m_1 * v_1.T*v_1 / 2
K = K_c + K_1

P_1 = Matrix([m_1 * g * p_1[1]])

# Lagrangian L=sum(K)-sum(P)
L = K_c + K_1 - P_1

# first term in the Euler-Lagrange exuation
partial_L_by_partial_x = L.jacobian(Matrix([x])).T

# inner term of the second part of the Euler-Lagrange exuation
partial_L_by_partial_xdot = L.jacobian(Matrix([xdot]))

# second term (overall, time derivative) in the Euler-Lagrange exuation
# applies the chain rule
d_inner_by_dt = partial_L_by_partial_xdot.jacobian(Matrix([x])) * xdot + partial_L_by_partial_xdot.jacobian(Matrix([xdot])) * xddot

# Euler-Lagrange exuation
lagrange_eq = partial_L_by_partial_x - d_inner_by_dt - Matrix([f,0])

# solve the lagrange exuation for xddot and simplify
print("Calculations take a while...")
r = sp.solvers.solve(simplify(lagrange_eq), Matrix([xddot]))
print("Calculations done, simplifying solutions...")

xddot_1 = simplify(r[xddot_1]);
xddot_2 = simplify(r[xddot_2]);
print("Done!")

# Solve for Coefficients for A * x = B where x = [ddth1 ddth2 ddt3]
A1 = lagrange_eq[0].coeff(sp.Symbol('xddot_1'))
A2 = lagrange_eq[0].coeff(sp.Symbol('xddot_2'))
A3 = lagrange_eq[1].coeff(sp.Symbol('xddot_1'))
A4 = lagrange_eq[1].coeff(sp.Symbol('xddot_2'))

# Multiply remaining terms by -1 to switch to other side of equation: A * x - B = 0 -> A * x = B 
# Multiplying means the input has to be multiplied by -1 again.
remainder = [(sp.Symbol('xddot_1'), 0), (sp.Symbol('xddot_2'), 0)]
B1 = -1 * lagrange_eq[0].subs(remainder)
B2 = -1 * lagrange_eq[1].subs(remainder)

# Calculate hdot
#First calculate L_g g(x)*u

# removing input as G matrix has been calculated

f1 = xdot_1
f2 = xddot_1
f3 = xdot_2
f4 = xddot_2

h1 = Emax - K_c[0]

g2 = 100/(100.0*sin(x_2)**2 - 600.0)
g4 = 50.0*sin(x_2)/(25.0*sin(x_2)**2 - 150.0)
G = sp.diff(h1,xdot_1)*g2+sp.diff(h1,xdot_2)*g4

print("Calculation hdot...")
h1dot = sp.diff(h1,x_1)*f1+sp.diff(h1,xdot_1)*f2+sp.diff(h1,x_2)*f3+sp.diff(h1,xdot_2)*f4
print("Done!")


replacements = [x_1,xdot_1,x_2,xdot_2] 
G1 = lambdify(replacements,G,'numpy')
K1 = lambdify(replacements,K,'numpy')
Kc = lambdify(replacements,K_c,'numpy')
h1dot = h1dot.subs({f:0})

H1 = lambdify(replacements,h1,'numpy')
H1dot = lambdify(replacements,h1dot,'numpy')

def asif(u_nominal,gamma,joint1, djoint1, joint2, djoint2):
    # objective function, same for all CBF-QP
    
    p = np.array([1])
    p = p.astype('float')
    q = -1*np.array([u_nominal])
    q = q.astype('float')
    g = np.array([G1(joint1, djoint1, joint2, djoint2)])
    g = g.astype('float')
    hasif = np.array([gamma*H1(joint1, djoint1, joint2, djoint2)+H1dot(joint1, djoint1, joint2, djoint2)])
    hasif = hasif.astype('float')
    u_filtered = solve_qp(p,q,g,hasif,solver = 'cvxopt')
    return u_filtered

A1 = lambdify(replacements, A1, "numpy")
A2 = lambdify(replacements, A2, "numpy")
A3 = lambdify(replacements, A3, "numpy")
A4 = lambdify(replacements, A4, "numpy")

replacements = [x_1,xdot_1,x_2,xdot_2, f] 

B1 = lambdify(replacements, B1, "numpy")
B2 = lambdify(replacements, B2, "numpy")

X1 = lambdify(replacements, xddot_1, "numpy")
X2 = lambdify(replacements, xddot_2, "numpy")
# Simulate System:
x0 = 0, 0, 2, 0  # th1, dth1, th2, dth2, th3, dth3
dt = 0.0001
sim_time = 1
time = np.arange(0, sim_time, dt)
sim_length = len(time)

# Desired states

# Control gains
Kp = np.array([0,0,0])
Kd = np.array([0,0,0])
use_asif = 1


# Initialize Arrays:
x1_vec = np.zeros(sim_length)
x1dot_vec = np.zeros(sim_length)

x2_vec = np.zeros(sim_length)
x2dot_vec = np.zeros(sim_length)

u_nom = np.zeros(sim_length)
u_cbf = np.zeros(sim_length)

p_nom = np.zeros(sim_length) #power
p_cbf = np.zeros(sim_length) #power

h_vec = np.zeros(sim_length) #kinetic energy
V_vec = np.zeros(sim_length) #potential energy
Ekin_vec = np.zeros(sim_length)
Ec_vec = np.zeros(sim_length)

hasif_vec = np.zeros(sim_length)
gasif_vec = np.zeros(sim_length)
# Evaluate Initial Conditions:
x1_vec[0] = x0[0]
x1dot_vec[0] = x0[1]

x2_vec[0] = x0[2]
x2dot_vec[0] = x0[3]

# Initialize A and B:
A = np.array([[0, 0], [0, 0]])
B = np.array([0, 0])

# Euler Integration:
for i in range(1, sim_length):
    print(i)
    #calculate energy
    h_vec[i-1] = H1(x1_vec[i-1],x1dot_vec[i-1],x2_vec[i-1],x2dot_vec[i-1])
    Ekin_vec[i-1] = K1(x1_vec[i-1],x1dot_vec[i-1],x2_vec[i-1],x2dot_vec[i-1])
    Ec_vec[i-1] = Kc(x1_vec[i-1],x1dot_vec[i-1],x2_vec[i-1],x2dot_vec[i-1])
    #calculate inputs:
    u_nom[i-1] = 0
    #u_cbf[i-1] = asif(u_nom[i-1],gamma,x1_vec[i-1],x1dot_vec[i-1],x2_vec[i-1],x2dot_vec[i-1])
    p = np.array([1])
    p = p.astype('float')
    q = -1*np.array([u_nom[i-1]])
    q = q.astype('float')
    g = np.array([G1(x1_vec[i-1],x1dot_vec[i-1],x2_vec[i-1],x2dot_vec[i-1])])
    g = g.astype('float')
    hasif = np.array([gamma*H1(x1_vec[i-1],x1dot_vec[i-1],x2_vec[i-1],x2dot_vec[i-1])+H1dot(x1_vec[i-1],x1dot_vec[i-1],x2_vec[i-1],x2dot_vec[i-1])])
    hasif = hasif.astype('float')
    u_cbf[i-1] = solve_qp(p,q,g,hasif,solver = 'cvxopt')
    
    gasif_vec[i-1] = G1(x1_vec[i-1],x1dot_vec[i-1],x2_vec[i-1],x2dot_vec[i-1])
    hasif_vec[i-1] = gamma*H1(x1_vec[i-1],x1dot_vec[i-1],x2_vec[i-1],x2dot_vec[i-1])+H1dot(x1_vec[i-1],x1dot_vec[i-1],x2_vec[i-1],x2dot_vec[i-1])
    
    
    # Evaluate Dynamics:
    A[0, 0] = A1(x1_vec[i-1],x1dot_vec[i-1],x2_vec[i-1],x2dot_vec[i-1])
    A[0, 1] = A2(x1_vec[i-1],x1dot_vec[i-1],x2_vec[i-1],x2dot_vec[i-1])
    A[1, 0] = A3(x1_vec[i-1],x1dot_vec[i-1],x2_vec[i-1],x2dot_vec[i-1])
    A[1, 1] = A4(x1_vec[i-1],x1dot_vec[i-1],x2_vec[i-1],x2dot_vec[i-1])
    if use_asif:
        B[0] = B1(x1_vec[i-1],x1dot_vec[i-1],x2_vec[i-1],x2dot_vec[i-1],u_cbf[i-1])
        B[1] = B2(x1_vec[i-1],x1dot_vec[i-1],x2_vec[i-1],x2dot_vec[i-1],u_cbf[i-1])
    else:
        print("this is not implemented")
    acc_1 = X1(x1_vec[i-1],x1dot_vec[i-1],x2_vec[i-1],x2dot_vec[i-1],u_cbf[i-1])
    acc_2 = X2(x1_vec[i-1],x1dot_vec[i-1],x2_vec[i-1],x2dot_vec[i-1],u_cbf[i-1])

    # Euler Step Integration:
    x1_vec[i] = x1_vec[i-1] + x1dot_vec[i-1] * dt
    x1dot_vec[i] = x1dot_vec[i-1] + acc_1 * dt

    x2_vec[i] = x2_vec[i-1] + x2dot_vec[i-1] * dt
    x2dot_vec[i] = x2dot_vec[i-1] + acc_2 * dt
    
# Plot states
    
fig, axs = plt.subplots(2, 2)
axs[0,0].plot(time,x1_vec, label = 'cart position')
axs[0,0].set_xlabel('time')
axs[0,0].set_ylabel('position [m]')
axs[0,0].set_title('cart position')
axs[0,1].plot(time,x1dot_vec, label = 'cart velocity')
axs[0,1].set_xlabel('time [s]')
axs[0,1].set_ylabel('velocity [m/s]')
axs[0,1].set_title('Joint 2, p-v [rad/s]')
axs[1,0].plot(time, x2_vec, label = 'angle')
axs[1,0].set_xlabel('time')
axs[1,0].set_ylabel('position [rad]')
axs[1,0].set_title('pendulum pos')
axs[1,1].plot(time, x2dot_vec, label = 'pendulum velocity')
axs[1,1].set_xlabel('time')
axs[1,1].set_ylabel('velocity [rad/s]')
axs[1,1].set_title('pendulum velocity')
plt.tight_layout()
plt.savefig('Position 2D pendulum, Em = [15,15]', dpi = 1000)
plt.show()

# Plot energy in the system
plt.plot(time, h_vec, label = 'Barrier function')
plt.plot(time, Ekin_vec, label = 'total kinetic energy')
plt.plot(time, Ec_vec, label = 'cart kinetic energy')
plt.xlabel('time')
plt.ylabel('Energy [J]')
plt.legend()
plt.title("Energy & barrier")
plt.show()

plt.plot(time, gasif_vec, label = 'lefthand')
plt.plot(time, hasif_vec, label = 'righthand')
plt.xlabel('time')
plt.ylabel('Energy [J]')
plt.legend()
plt.title("Barrier function constraint")
plt.show()


