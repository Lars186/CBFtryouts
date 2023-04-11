# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 15:55:23 2023

@author: lesse
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy

from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import sympy.utilities.lambdify as lambdify
from qpsolvers import solve_qp

# Pendulum Parameters:
m_1 = 1  # kg
m_2 = 1  # kg
m_3 = 1  # kg
l_1 = 1  # m
l_2 = 1  # m
l_3 = 1  # m
g = 9.8  # m/s^2


# Create Symbols for Time:
t = sympy.Symbol('t')  # Creates symbolic variable t
u1 = sympy.Symbol('u1')
u2 = sympy.Symbol('u2')
u3 = sympy.Symbol('u3')
# Create Generalized Coordinates as a function of time: q = [theta_1, theta_2, theta_3]
th1 = sympy.Function('th1')(t)
th2 = sympy.Function('th2')(t)   
th3 = sympy.Function('th3')(t)   

# Create derivatives as symbolics
dth1,dth2,dth3,ddth1,ddth2,ddth3 = sympy.symbols('dth1,dth2,dth3,ddth1,ddth2,ddth3')
  
# Position Equation: r = [x, y]
r1 = np.array([l_1 * sympy.sin(th1), -l_1 * sympy.cos(th1)])  # Position of first pendulum
r2 = np.array([l_2 * sympy.sin(th2) + r1[0], -l_2 * sympy.cos(th2) + r1[1]])  # Position of second pendulum
r3 = np.array([l_3 * sympy.sin(th3) + r2[0], -l_3 * sympy.cos(th3) + r2[1]])  # Position of third pendulum

# Velocity Equation: d/dt(r) = [dx/dt, dy/dt]
v1 = np.array([r1[0].diff(t), r1[1].diff(t)])  # Velocity of first pendulum
v2 = np.array([r2[0].diff(t), r2[1].diff(t)])  # Velocity of second pendulum
v3 = np.array([r3[0].diff(t), r3[1].diff(t)])  # Velocity of third pendulum

# Energy Equations:
T1, T2, T3 = (1/2 * m_1 * np.dot(v1, v1)), (1/2 * m_2 * np.dot(v2, v2)) , (1/2 * m_3 * np.dot(v3, v3))  # Kinetic Energy
T = T1 + T2 + T3 #kinetic
V1, V2, V3 = m_1 * g * r1[1], m_2 * g * r2[1] , m_3 * g * r3[1] # Potential Energy
V = V1 + V2 + V3
L = T - V  # Lagrangian

# Lagrange Terms:
dL_dth1 = L.diff(th1)
dL_dth1_dt = L.diff(th1.diff(t)).diff(t)
dL_dth2 = L.diff(th2)
dL_dth2_dt = L.diff(th2.diff(t)).diff(t)
dL_dth3 = L.diff(th3)
dL_dth3_dt = L.diff(th3.diff(t)).diff(t)

# Euler-Lagrange Equations: dL/dq - d/dt(dL/ddq) - u = 0
th1_eqn = dL_dth1 - dL_dth1_dt - u1
th2_eqn = dL_dth2 - dL_dth2_dt - u2
th3_eqn = dL_dth3 - dL_dth3_dt - u3

# Replace Time Derivatives and Functions with Symbolic Variables:
replacements = [(th1.diff(t).diff(t), sympy.Symbol('ddth1')), (th1.diff(t), sympy.Symbol('dth1')), (th1, sympy.Symbol('th1')), 
                (th2.diff(t).diff(t), sympy.Symbol('ddth2')), (th2.diff(t), sympy.Symbol('dth2')), (th2, sympy.Symbol('th2')),
                (th3.diff(t).diff(t), sympy.Symbol('ddth3')), (th3.diff(t), sympy.Symbol('dth3')), (th3, sympy.Symbol('th3'))]

th1_eqn = th1_eqn.subs(replacements)
th2_eqn = th2_eqn.subs(replacements)
th3_eqn = th3_eqn.subs(replacements)

r1 = r1[0].subs(replacements), r1[1].subs(replacements)
r2 = r2[0].subs(replacements), r2[1].subs(replacements)
r3 = r3[0].subs(replacements), r3[1].subs(replacements)

T1,T2,T3 = T1.subs(replacements),T2.subs(replacements),T3.subs(replacements)
V1,V2,V3 = V1.subs(replacements),V2.subs(replacements),V3.subs(replacements)

# Simplfiy then Force SymPy to Cancel factorization:
# Set eqn equal to the control torques tau
th1_eqn = sympy.simplify(th1_eqn)
th2_eqn = sympy.simplify(th2_eqn)
th3_eqn = sympy.simplify(th3_eqn)
th1_eqn = th1_eqn.cancel()
th2_eqn = th2_eqn.cancel()
th3_eqn = th3_eqn.cancel()

# if a solution already exists in workspace do not recalculate
if not 'sol1' in locals():
    ## Create control barrier functions
    print("calculating solutions to DEs, this may take some time...")
    sols = sympy.solvers.solve([th1_eqn,th2_eqn,th3_eqn],ddth1,ddth2,ddth3)
    print("Solutions found, simplifying")
    sol1 = sols[ddth1].simplify()
    sol2 = sols[ddth2].simplify()
    sol3 = sols[ddth3].simplify()
    
    sol1 = sol1.subs(replacements)
    sol2 = sol2.subs(replacements)
    sol3 = sol3.subs(replacements)
    f1,f2,f3,f4,f5,f6 = dth1,sol1,dth2,sol2,dth3,sol3
else:
    print("Skiping solving DEs")  


# H = Em - Ekin, Em is constant so when differentiating this can be ommited
h1,h2,h3 = -T1,-T2,-T3
print("Calculating input dependent part hdot...")

#g matrix of dynamics from sol, obtained by looking at the solution to the differential equations
g1 = 0
g2 = [-5.0*sympy.cos(th2 - th3)**2 + 10.0, -10.0*sympy.cos(th1 - th2) + 5.0*sympy.cos(th1 - th3)*sympy.cos(th2 - th3), 10.0*sympy.cos(th1 - th2)*sympy.cos(th2 - th3) - 10.0*sympy.cos(th1 - th3)]
g4 = [-10.0*sympy.cos(th1 - th2) + 5.0*sympy.cos(th1 - th3)*sympy.cos(th2 - th3), -5.0*sympy.cos(th1 - th3)**2 + 15.0, 10.0*sympy.cos(th1 - th2)*sympy.cos(th1 - th3) - 15.0*sympy.cos(th2 - th3)]
g5 = 0
g6 = [10.0*sympy.cos(th1 - th2)*sympy.cos(th2 - th3) - 10.0*sympy.cos(th1 - th3), 10.0*sympy.cos(th1 - th2)*sympy.cos(th1 - th3) - 15.0*sympy.cos(th2 - th3), -20.0*sympy.cos(th1 - th2)**2 + 30.0]
g2 = np.array([g2[0].subs(replacements),g2[1].subs(replacements),g2[2].subs(replacements)])
g4 = np.array([g4[0].subs(replacements),g4[1].subs(replacements),g4[2].subs(replacements)])
g6 = np.array([g6[0].subs(replacements),g6[1].subs(replacements),g6[2].subs(replacements)])

G = np.array([[sympy.diff(h1,dth1)*g2+sympy.diff(h1,dth2)*g4+sympy.diff(h1,dth3)*g6,
               sympy.diff(h2,dth1)*g2+sympy.diff(h2,dth2)*g4+sympy.diff(h2,dth3)*g6,
               sympy.diff(h3,dth1)*g2+sympy.diff(h3,dth2)*g4+sympy.diff(h3,dth3)*g6,]]).reshape(3,3)
print("G calculations done!")
# removing input as G matrix has been calculated
f2,f4,f6 = f2.subs({u1:0,u2:0,u3:0}),f4.subs({u1:0,u2:0,u3:0}),f6.subs({u1:0,u2:0,u3:0})

print("Calculation hdot...")
h1dot = sympy.diff(h1,th1)*f1+sympy.diff(h1,dth1)*f2+sympy.diff(h1,th2)*f3+sympy.diff(h1,dth2)*f4+sympy.diff(h1,th3)*f5+sympy.diff(h1,dth2)*f6
h2dot = sympy.diff(h2,th1)*f1+sympy.diff(h2,dth1)*f2+sympy.diff(h2,th2)*f3+sympy.diff(h2,dth2)*f4+sympy.diff(h2,th3)*f5+sympy.diff(h2,dth3)*f6
h3dot = sympy.diff(h3,th1)*f1+sympy.diff(h3,dth1)*f2+sympy.diff(h3,th2)*f3+sympy.diff(h3,dth2)*f4+sympy.diff(h3,th3)*f5+sympy.diff(h3,dth3)*f6
print("done!")
# Solve for Coefficients for A * x = B where x = [ddth1 ddth2 ddth3]
A1 = th1_eqn.coeff(sympy.Symbol('ddth1'))
A2 = th1_eqn.coeff(sympy.Symbol('ddth2'))
A3 = th1_eqn.coeff(sympy.Symbol('ddth3'))

A4 = th2_eqn.coeff(sympy.Symbol('ddth1'))
A5 = th2_eqn.coeff(sympy.Symbol('ddth2'))
A6 = th2_eqn.coeff(sympy.Symbol('ddth3'))

A7 = th3_eqn.coeff(sympy.Symbol('ddth1'))
A8 = th3_eqn.coeff(sympy.Symbol('ddth2'))
A9 = th3_eqn.coeff(sympy.Symbol('ddth3'))

# Multiply remaining terms by -1 to switch to other side of equation: A * x - B = 0 -> A * x = B 
# Multiplying means the input has to be multiplied by -1 again.
remainder = [(sympy.Symbol('ddth1'), 0), (sympy.Symbol('ddth2'), 0), (sympy.Symbol('ddth3'), 0)]
B1 = -1 * th1_eqn.subs(remainder)
B2 = -1 * th2_eqn.subs(remainder)
B3 = -1 * th3_eqn.subs(remainder)

# Generate Lambda Functions for A and B and stuff we want to plot
replacements = (sympy.Symbol('th1'), sympy.Symbol('dth1'), sympy.Symbol('th2'), sympy.Symbol('dth2'), sympy.Symbol('th3'), sympy.Symbol('dth3'))
A1 = lambdify(replacements, A1, "numpy")
A2 = lambdify(replacements, A2, "numpy")
A3 = lambdify(replacements, A3, "numpy")

A4 = lambdify(replacements, A4, "numpy")
A5 = lambdify(replacements, A5, "numpy")
A6 = lambdify(replacements, A6, "numpy")

A7 = lambdify(replacements, A7, "numpy")
A8 = lambdify(replacements, A8, "numpy")
A9 = lambdify(replacements, A9, "numpy")

r1 = lambdify(replacements, r1, "numpy")
r2 = lambdify(replacements, r2, "numpy")
r3 = lambdify(replacements, r3, "numpy")

H1 = lambdify(replacements, h1dot, "numpy")
H2 = lambdify(replacements, h2dot, "numpy")
H3 = lambdify(replacements, h3dot, "numpy")
G1 = lambdify(replacements, G[0,0], "numpy")
G2 = lambdify(replacements, G[0,1], "numpy")
G3 = lambdify(replacements, G[0,2], "numpy")
G4 = lambdify(replacements, G[1,0], "numpy")
G5 = lambdify(replacements, G[1,1], "numpy")
G6 = lambdify(replacements, G[1,2], "numpy")
G7 = lambdify(replacements, G[2,0], "numpy")
G8 = lambdify(replacements, G[2,1], "numpy")
G9 = lambdify(replacements, G[2,2], "numpy")

T1,T2,T3 = lambdify(replacements, T1, "numpy"),lambdify(replacements, T2, "numpy"),lambdify(replacements, T3, "numpy")
V1,V2,V3 = lambdify(replacements, V1, "numpy"),lambdify(replacements, V2, "numpy"),lambdify(replacements, V3, "numpy")

replacements = (sympy.Symbol('th1'), sympy.Symbol('dth1'), sympy.Symbol('th2'), sympy.Symbol('dth2')
                ,sympy.Symbol('th3'), sympy.Symbol('dth3'),sympy.Symbol('u1'),sympy.Symbol('u2')
                ,sympy.Symbol('u3'))

B1 = lambdify(replacements, B1, "numpy")
B2 = lambdify(replacements, B2, "numpy")
B3 = lambdify(replacements, B3, "numpy")

# create weights for the CBF, sum of weights equals 1
w1,w2,w3 = 1/3,1/3,1/3

def asif(u_nominal,gamma,joint1, djoint1, joint2, djoint2,joint3, djoint3,E):
    # Rewrite matrices for the solve_qp function    
    p = np.array([[w1,0.0,0.0],[0.0,w2,0.0],[0.0,0.0,w3]]).reshape(3,3)
    q = -2*np.array([w1*u_nominal[0],w2*u_nominal[1],w3*u_nominal[2]])
    q = q.astype('float')
    g = np.array([G1(joint1, djoint1, joint2, djoint2,joint3,djoint3),G2(joint1, djoint1, joint2, djoint2,joint3,djoint3),G3(joint1, djoint1, joint2, djoint2,joint3,djoint3),
                  G4(joint1, djoint1, joint2, djoint2,joint3,djoint3),G5(joint1, djoint1, joint2, djoint2,joint3,djoint3),G6(joint1, djoint1, joint2, djoint2,joint3,djoint3),
                  G7(joint1, djoint1, joint2, djoint2,joint3,djoint3),G8(joint1, djoint1, joint2, djoint2,joint3,djoint3),G9(joint1, djoint1, joint2, djoint2,joint3,djoint3)]).reshape(3,3)
    g = g.astype('float')
    Hasif1 = gamma*(Emax[0]-E[0]) + H1(joint1, djoint1, joint2, djoint2,joint3,djoint3)
    Hasif2 = gamma*(Emax[1]-E[1]) + H2(joint1, djoint1, joint2, djoint2,joint3,djoint3)
    Hasif3 = gamma*(Emax[2]-E[2]) + H3(joint1, djoint1, joint2, djoint2,joint3,djoint3)
    h = np.array([Hasif1,Hasif2,Hasif3])
    h = h.astype('float')
      
    u_filtered = solve_qp(p,q,g,h,solver = 'quadprog')
    return u_filtered

def _h(E):
    h1 = Emax[0]-E[0]
    h2 = Emax[1]-E[1]
    h3 = Emax[2]-E[2]
    return np.array([h1,h2,h3])


# Simulate System:
x0 = 2, 0, 2, 0, 2, 0  # th1, dth1, th2, dth2, th3, dth3
dt = 0.0001
sim_time = 5
time = np.arange(0, sim_time, dt)
sim_length = len(time)

# Desired thetas
th1_des  = np.pi/2
th2_des  = 4*np.pi/4
th3_des  = np.pi/2
dth1_des = 0
dth2_des = 0
dth3_des = 0

# Control gains
Kp = np.array([0,0,0])
Kd = np.array([0,0,0])
use_asif = 1

# CBF parameters
gamma = 2
Emax = np.array([20,20,20])

# Initialize Arrays:
th1_vec = np.zeros(sim_length)
dth1_vec = np.zeros(sim_length)

th2_vec = np.zeros(sim_length)
dth2_vec = np.zeros(sim_length)

th3_vec = np.zeros(sim_length)
dth3_vec = np.zeros(sim_length)

x1_vec = np.zeros(sim_length)
y1_vec = np.zeros(sim_length)

x2_vec = np.zeros(sim_length)
y2_vec = np.zeros(sim_length)

x3_vec = np.zeros(sim_length)
y3_vec = np.zeros(sim_length)

u_nom = np.zeros((sim_length,3))
u_cbf = np.zeros((sim_length,3))

p_nom = np.zeros((sim_length,3)) #power
p_cbf = np.zeros((sim_length,3)) #power

T_vec = np.zeros((sim_length,3)) #kinetic energy
V_vec = np.zeros((sim_length,3)) #potential energy

left = np.zeros((sim_length,3))
right = np.zeros((sim_length,3))
g_vec = np.zeros((sim_length,3,3))

# Evaluate Initial Conditions:
th1_vec[0] = x0[0]
dth1_vec[0] = x0[1]

th2_vec[0] = x0[2]
dth2_vec[0] = x0[3]

th3_vec[0] = x0[4]
dth3_vec[0] = x0[5]

x1_vec[0], y1_vec[0] = r1(x0[0], x0[1], x0[2], x0[3], x0[4], x0[5])
x2_vec[0], y2_vec[0] = r2(x0[0], x0[1], x0[2], x0[3], x0[4], x0[5])
x3_vec[0], y3_vec[0] = r3(x0[0], x0[1], x0[2], x0[3], x0[4], x0[5])

# Initialize A and B:
A = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
B = np.array([0, 0, 0])

# Euler Integration:
for i in range(1, sim_length):
    print(i)
    #calculate energy
    T_vec[i-1] = T1(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1]),T2(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1]),T3(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1])
    V_vec[i-1] = V1(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1]),V2(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1]),V3(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1])
    #calculate inputs:
    u_nom[i-1,0] = -1*Kp[0]*(th1_des-th1_vec[i-1]) - Kd[0]*(dth1_des-dth1_vec[i-1])
    u_nom[i-1,1] = -1*Kp[1]*(th2_des-th2_vec[i-1]) - Kd[1]*(dth2_des-dth2_vec[i-1])
    u_nom[i-1,2] = -1*Kp[2]*(th3_des-th3_vec[i-1]) - Kd[2]*(dth3_des-dth3_vec[i-1])
    
    u_cbf[i-1,:] = asif(u_nom[i-1],gamma,th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1],T_vec[i-1])
      
    # Evaluate Dynamics:
    A[0, 0] = A1(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1])
    A[0, 1] = A2(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1])
    A[0, 2] = A3(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1])

    A[1, 0] = A4(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1])
    A[1, 1] = A5(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1])
    A[1, 2] = A6(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1])

    A[2, 0] = A7(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1])
    A[2, 1] = A8(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1])
    A[2, 2] = A9(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1])
    if use_asif:
        B[0] = B1(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1], u_cbf[i-1,0], u_cbf[i-1,1], u_cbf[i-1,1])
        B[1] = B2(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1],u_cbf[i-1,0], u_cbf[i-1,1], u_cbf[i-1,1])
        B[2] = B3(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1],u_cbf[i-1,0], u_cbf[i-1,1], u_cbf[i-1,1])
    else:
        B[0] = B1(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1], u_nom[i-1,0], u_nom[i-1,1], u_nom[i-1,2])
        B[1] = B2(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1],u_nom[i-1,0], u_nom[i-1,1], u_nom[i-1,2])
        B[2] = B3(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1],u_nom[i-1,0], u_nom[i-1,1], u_nom[i-1,2])

    [ddth1, ddth2, ddth3] = np.linalg.solve(A, B)

    # Euler Step Integration:
    th1_vec[i] = th1_vec[i-1] + dth1_vec[i-1] * dt
    dth1_vec[i] = dth1_vec[i-1] + ddth1 * dt

    th2_vec[i] = th2_vec[i-1] + dth2_vec[i-1] * dt
    dth2_vec[i] = dth2_vec[i-1] + ddth2 * dt

    th3_vec[i] = th3_vec[i-1] + dth3_vec[i-1] * dt
    dth3_vec[i] = dth3_vec[i-1] + ddth3 * dt

    # Visualisation States:
    x1_vec[i], y1_vec[i] = r1(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1])
    x2_vec[i], y2_vec[i] = r2(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1])
    x3_vec[i], y3_vec[i] = r3(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1])
    
    p_nom[i-1,0],p_nom[i-1,1],p_nom[i-1,2]=-1*u_nom[i-1,0]*dth1_vec[i-1],-1*u_nom[i-1,1]*dth2_vec[i-1],-1*u_nom[i-1,2]*dth3_vec[i-1]
    p_cbf[i-1,0],p_cbf[i-1,1],p_cbf[i-1,2]=-1*u_cbf[i-1,0]*dth1_vec[i-1],-1*u_cbf[i-1,1]*dth2_vec[i-1],-1*u_cbf[i-1,2]*dth3_vec[i-1]

    g_vec[i] = np.array([G1(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1]),G2(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1]),G3(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1]),
                         G4(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1]),G5(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1]),G6(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1]),
                         G7(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1]),G8(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1]),G9(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1])]).reshape(3,3)
    
    left[i]=np.matmul(g_vec[i],[u_cbf[i-1,0], u_cbf[i-1,1], u_cbf[i-1,1]])
    
    Hasif1 = gamma*(Emax[0]-T_vec[i-1,0]) + H1(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1])
    Hasif2 = gamma*(Emax[1]-T_vec[i-1,1]) + H2(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1])
    Hasif3 = gamma*(Emax[2]-T_vec[i-1,2]) + H3(th1_vec[i-1], dth1_vec[i-1], th2_vec[i-1], dth2_vec[i-1], th3_vec[i-1], dth3_vec[i-1])
    right[i] = np.array([Hasif1,Hasif2,Hasif3])
    
    
# Plot all theta values over time:
plt.plot(time, th1_vec, label = 'theta1')
plt.plot(time, th2_vec, label = 'theta2')
plt.plot(time, th3_vec,label = 'theta1')
plt.xlabel('time')
plt.ylabel('Angle [rad]')
plt.legend()
plt.title("Angles over time")
plt.savefig('angles over time', dpi = 1000)
plt.show()
plt.show()

# Plot energy in the system
plt.plot(time, T_vec[:,0], label = 'Kinetic energy joint 1')
plt.plot(time, T_vec[:,1], label = 'Kinetic energy joint 2')
plt.plot(time, T_vec[:,2],label = 'Kinetic energy joint 3')
#plt.plot(time, V_vec[:,0], label = 'Potential energy joint 1')
#plt.plot(time, V_vec[:,1], label = 'Potential energy joint 2')
#plt.plot(time, V_vec[:,2],label = 'Potential energy joint 3')
plt.xlabel('time')
plt.ylabel('Energy [J]')
plt.legend()
plt.title("Kinetic")
plt.savefig('Energy over time', dpi = 1000)
plt.show()

# pv plot of all angles
plt.plot(th1_vec, dth1_vec, label = 'joint 1')
plt.plot(th2_vec, dth2_vec, label = 'joint 2')
plt.plot(th3_vec, dth3_vec,label = 'joint 3')
plt.xlabel('position [rad]')
plt.ylabel('Velocity [rad/s]')
plt.legend()
plt.title("p-v plot of all joints")
plt.savefig('PV', dpi = 1000)
plt.show()


# =============================================================================
# # plot effect of CBF
# plt.plot(time, u_nom[:,0], label = 'nominal joint 1')
# plt.plot(time, u_nom[:,1], label = 'nominal joint 2')
# plt.plot(time, u_nom[:,2], label = 'nominal joint 3')
# plt.plot(time, u_cbf[:,0], label = 'cbf joint 1')
# plt.plot(time, u_cbf[:,1], label = 'cbf joint 2')
# plt.plot(time, u_cbf[:,2], label = 'cbf joint 3')
# plt.xlabel('time')
# plt.ylabel('Input [rad]')
# plt.legend()
# plt.title("input")
# plt.show()
# =============================================================================

# plot effect of CBF
plt.plot(time, p_nom[:,0], label = 'joint 1')
plt.plot(time, p_nom[:,1], label = 'joint 2')
plt.plot(time, p_nom[:,2], label = 'joint 3')
plt.plot(time, (p_nom[:,0]+p_nom[:,1]+p_nom[:,2]), label = 'total power')
plt.xlabel('time')
plt.ylabel('Power injected [watts]')
plt.legend()
plt.title("Power nominal")
plt.savefig('Power nominal', dpi = 1000)
plt.show()

# plot effect of CBF
plt.plot(time, p_cbf[:,0], label = 'joint 1')
plt.plot(time, p_cbf[:,1], label = 'joint 2')
plt.plot(time, p_cbf[:,2], label = 'joint 3')
plt.plot(time, (p_cbf[:,0]+p_cbf[:,1]+p_cbf[:,2]), label = 'total power')
plt.xlabel('time')
plt.ylabel('Power injected [watts]')
plt.legend()
plt.title("Power CBF")
plt.savefig('Power cbf', dpi = 1000)
plt.show()

# plot constraints
plt.plot(time, left[:,0], label = 'Lefthand joint 1')
plt.plot(time, right[:,0], label = 'righthandthand joint 1')
plt.xlabel('time')
plt.ylabel('Power injected [watts]')
plt.legend()
plt.title("Constraint joint 1")
plt.show()

# plot constraints
plt.plot(time, left[:,1], label = 'Lefthand joint 2')
plt.plot(time, right[:,1], label = 'righthandthand joint 2')
plt.xlabel('time')
plt.ylabel('Power injected [watts]')
plt.legend()
plt.title("Constraints joint 2")
plt.show()

# plot constraints
plt.plot(time, left[:,2], label = 'Lefthand joint 3')
plt.plot(time, right[:,2], label = 'righthandthand joint 3')
plt.xlabel('time')
plt.ylabel('Power injected [watts]')
plt.legend()
plt.title("Constraint joint 3")
plt.show()

# Create Animation:
# Setup Figure: Initialize Figure / Axe Handles
fig, ax = plt.subplots()
p1, = ax.plot([], [], color='black', linewidth=2)
p2, = ax.plot([], [], color='black', linewidth=2)
p3, = ax.plot([], [], color='black', linewidth=2)
lb, ub = -5, 5
ax.axis('equal')
ax.set_xlim([lb, ub])
ax.set_xlabel('X')  # X Label
ax.set_ylabel('Y')  # Y Label
ax.set_title('Triple Pendulum Simulation:')
video_title = "simulation"

# Setup Animation Writer:
FPS = 20
sample_rate = int(1 / (dt * FPS))
dpi = 300
writerObj = FFMpegWriter(fps=FPS)

# Initialize Patch: Pendulum 1 and 2
pendulum_1 = Circle((0, 0), radius=0.1, color='cornflowerblue', zorder=10)
pendulum_2 = Circle((0, 0), radius=0.1, color='cornflowerblue', zorder=10)
pendulum_3 = Circle((0, 0), radius=0.1, color='cornflowerblue', zorder=10)
ax.add_patch(pendulum_1)
ax.add_patch(pendulum_2)
ax.add_patch(pendulum_3)

# Draw Static Objects:
pin_joint = Circle((0, 0), radius=0.05, color='black', zorder=10)
ax.add_patch(pin_joint)

# Plot and Create Animation:
with writerObj.saving(fig, video_title+".mp4", dpi):
    for i in range(0, sim_length, sample_rate):
        # Draw Pendulum Arm:
        x_pendulum_arm = [0, x1_vec[i], x2_vec[i], x3_vec[i]]
        y_pendulum_arm = [0, y1_vec[i], y2_vec[i], y3_vec[i]]
        p1.set_data(x_pendulum_arm, y_pendulum_arm)
        # Update Pendulum Patches:
        pendulum_1_center = x1_vec[i], y1_vec[i]
        pendulum_2_center = x2_vec[i], y2_vec[i]
        pendulum_3_center = x3_vec[i], y3_vec[i]

        pendulum_1.center = pendulum_1_center
        pendulum_2.center = pendulum_2_center
        pendulum_3.center = pendulum_3_center
        # Update Drawing:
        fig.canvas.draw()
        # Grab and Save Frame:
        writerObj.grab_frame()
from IPython.display import Video
Video("/work/"+video_title+".mp4", embed=True, width=640, height=480)   