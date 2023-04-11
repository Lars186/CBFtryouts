# Readme
Creation date: 28-02-23
Last edited: 11-04-23

This project contains some 1D insights on using control barrier functions. It is recommended to look at 3dof.py as this is the most complete example. Pendulum.py constains input constraints and can be interesting to look at as well. For all multiple dimension projects calculations are done symbolically using Sympy.  1DOF dynamics have been calculated by hand.


## multiple DOF
- 3dof.py is a 3DOF pendulum hanging down. The barrier limits kinetic energy for every joint.
- pendcart_hvel.py is an underactuated case which limits total kinetic energy. This project is abandoned as quadratic programs quickly become unfeasible.


## 1DOF
- Pendulum.py
- Movingmass.py contains position constraints on a simple 1D moving mass, the input is the force.
- movingmassMultiConstraint.py has constrains velocity and position by using 2 ASICs. This is not a nice way of solving
- EnergyConstraints.py limits kinetic energy for the moving mass.


