#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Affinely parametrized linear BVP:
     - div( lambda(mu) * grad(u) ) + w * grad(u) = f  in domain
                                       u = g  on bdry dirichlet
                         - lambda(mu) nabla(u).n = 0 on bdry Neumann
with w: given velocity field

Input parameter (scalar parameter): mu (the diffusivity coeff.)
    
Goal: Solve this BVP by an offline-online strategy based on a POD.
 
'''
#import system
from dolfin import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigh
import time
import random
import numpy.linalg as npl
import scipy
import scipy.linalg   
import math
from mpl_toolkits.mplot3d import axes3d
#

# The PDE parameter: diffusivity lambda(mu)
def Lambda(mu):
    mu0 = 0.7
    #    return  mu + mu0 # affine case
    return np.exp(mu0*(mu+1.)) # non-affine case

# Function to compute the RB dimension (= Nrb)
def energy_number(epsilon_POD,lam):
    # lam: eigenvalues table
    # return the eignvalue number corresponding to energy_ratio
    index_min = 0; s = 0.;s1=np.sum(lam)
    for i in range(len(lam)):
        if s < s1*(1-epsilon_POD):
            s += lam[i]
            index_min = index_min + 1
    return index_min

# Dirichlet boundary conditions
tol_bc = 1.e-10
def u_bdry_0(x, on_boundary):
    return bool(on_boundary and (near(x[0], 0, tol_bc)))
def u_bdry_1(x, on_boundary):
    return bool(on_boundary and (near(x[0], 1, tol_bc)))



###################################################
#    MAIN program 
###################################################
##############
# Offline phase
##############
# Physical and numerical parameters
# Mesh and function spaces
#system.os("clear")
NP =  100; print('Number of mesh points NP = ', NP)
mesh = UnitSquareMesh(NP,NP)
k = 2 ; print('Order of the Lagrange FE k = ', k)
V = FunctionSpace(mesh, "CG", int(k))
V_vec = VectorFunctionSpace(mesh, "CG", int(k))
NN = V.dim(); print('Resulting number of nodes NN = ', NN)
coordinates = mesh.coordinates()
# Trial and test function
u, v = TrialFunction(V), TestFunction(V)

# Snapshots number
print('How many snapshots do I compute ? ')
M = int(input())

# Range of mu values
mu_min = 1.0; mu_max = 10. # MAY BE CHANGED
print('Range values for mu: [',mu_min,',',mu_max,']')
mu = np.linspace(mu_min,mu_max,M)
ans = input('type any key to resume...')

# Plot of the parameter space
Param =  np.zeros(len(mu))
for i in range(len(mu)):
    Param[i] = Lambda(mu[i])
print("Param=",Param)
fig = plt.figure()
ax = fig.gca() 
ax.scatter(mu, Param) 
plt.title("Parameter space")
ax.set_xlabel('Input parameter mu')
ax.set_ylabel('Lambda(mu)')
plt.legend()
plt.show()

# Velocity field
vel_amp = 1e+2; print('vel_amp =',vel_amp)
vel_exp = Expression(('(1.+abs(cos(2*pi*x[0])))', 'sin(2*pi/0.2*x[0])'), element = V.ufl_element())
#vel_exp = Expression(('0.', '0.'), element = V.ufl_element())
vel = vel_amp * interpolate(vel_exp,V_vec)
#p=plot(vel,title='The velocity field')
#p.set_cmap("rainbow"); plt.colorbar(p); plt.show()

print("Compute and plot Peclet numbers...") 
print("   Peclet number for max value of mu")
mu_max = Lambda(np.max(mu))
Pe_min = sqrt(dot(vel,vel))/(mu_max)
Pe_func_min = project(Pe_min, V)
Pe_vec_min = Pe_func_min.vector().get_local()
print("min_Pe_min",min(Pe_vec_min))
print("max_Pe_min",max(Pe_vec_min))
p=plot(Pe_func_min.leaf_node(), title="Peclet number for max value of mu")
p.set_cmap("rainbow")# ou 'viridis
plt.colorbar(p); plt.show()

print("    Peclet number for min value of mu")
mu_min = Lambda(np.min(mu))
Pe_max = sqrt(dot(vel,vel))/(mu_min)
Pe_func_max = project(Pe_max, V)
Pe_vec_max = Pe_func_max.vector().get_local()
print("min_Pe_max",min(Pe_vec_max))
print("max_Pe_max",max(Pe_vec_max))
p=plot(Pe_func_max.leaf_node(), title="Peclet number for min value of mu")
p.set_cmap("rainbow")# ou 'viridis
plt.colorbar(p); plt.show()

import sys
# RHS of the PDE model
f_exp = Expression('1E+03 * exp( -( abs(x[0]-0.5) + abs(x[1]-0.5) ) / 0.1 )', element = V.ufl_element()) # Gaussian
f = interpolate(f_exp,V)

print('#')
print('# Computation of the M snapshots')
print('#')
Usnap = np.zeros((M,NN)) # Snaphots matrix
uh = np.zeros(NN)
t_0 =  time.time()
for m in range(M):
    print('snapshot #',m,' : mu = ',mu[m])
    diffus = Lambda(mu[m])
    print('snapshot #',m,' : Lambda(mu) = ',diffus)
    # Variational formulation
    F = diffus * dot(grad(v),grad(u)) * dx + v * dot(vel, grad(u)) * dx - f * v * dx
    # Stabilization of the advection term by SUPG 
    r = - diffus * div( grad(u) ) + dot(vel, grad(u)) - f #residual
    vnorm = sqrt( dot(vel, vel) )
    h = MaxCellEdgeLength(mesh)
    delta = h / (2.0*vnorm)
    F += delta * dot(vel, grad(v)) * r * dx
    # Create bilinear and linear forms
    a = lhs(F); L = rhs(F)
    # Dirichlet boundary conditions
    u_diri0_exp = Expression('1.', degree=u.ufl_element().degree())
    bc0 = DirichletBC(V, u_diri0_exp, u_bdry_0)
    # Solve the problem
    u_mu = Function(V)
    solve(a == L, u_mu,bc0)

    # Buid up the snapshots matrices U & nabla(U)
    uh = u_mu.vector().get_local()[:]
    Usnap[m, :] = uh # dim. M x NN
#
# Plot of the manifold in 3D: three arbitrary components
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 
 
# X, Y, Z = axes3d.get_test_data(0.05)
# cset = ax.contour(X, Y, Z, 16, extend3d=True)
# ax.clabel(cset, fontsize=9, inline=1)
# plt.show()
X = Usnap[:,round(NN/3)]; Y = Usnap[:,round(NN/2)]; Z = Usnap[:,round(2*NN/3)]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X, Y, Z, label='manifold') 
ax.scatter(X, Y,Z, label='manifold')
plt.title("A 3D restriction of the manifold")
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
plt.show()
#
# Transpose the snapshots matrix to be of size NN x M
Usnap = Usnap.transpose()

# Assemble of the rigitdiy matrix (FE problem in V_h)
f_exp1 = Expression('0.0', element = V.ufl_element())
f1 = interpolate(f_exp1,V)
u1, v1 = TrialFunction(V), TestFunction(V)
F1 =  dot(grad(v1),grad(u1)) * dx +  u1 * v1 * dx + f1 * v1 * dx
a1 = lhs(F1); L1 = rhs(F1)
# Assemble & get the matrix NN x NN
A_ass1, F1 = assemble_system(a1, L1)
# For L2 norm we have:
A_NN1 = np.identity(NN)

####################  POD  method ###################
# Computation of the correlation matrix for the L2 norm
C = np.dot(Usnap.transpose(),Usnap)

# Solve the eigenvalue problem C.w = lambda.w
Eigen_val, Eigen_vec = np.linalg.eig(C)

# Computation of the left singular vector from the eigenvectors of C
Xi = np.zeros((len(Usnap),len(Eigen_vec)))
for i in range(len(Eigen_vec)):
    Xi[:,i] = (1/(sqrt(Eigen_val[i])))*np.dot(Usnap, Eigen_vec[:,i])

# Normalization of the eigenvalues 
s_eigen = np.sum(Eigen_val)
Eigen_val_normalized = np.zeros(len(Eigen_val))
for i in range(len(Eigen_val_normalized)):
    Eigen_val_normalized[i] = (Eigen_val[i])/s_eigen

# Plot of the eigenvalues
Decay = np.arange(len(Eigen_val))
fig = plt.figure()
ax = fig.gca() 
ax.plot(Decay, abs(Eigen_val), label='Eigenvalues',color='r') 
#plt.bar(Decay, DD,width=0.0001)
plt.title("Eigenvalues (ordered in decreasing way)")
ax.set_xlabel('Eigenvalue index')
ax.set_ylabel('Eigenvalue')
plt.yscale("log")
plt.legend()
# Plot in bar chart
width = 0.5
p =plt.bar(Decay,Eigen_val, width, color='b');
plt.title('The M eigenvalues');plt.ylabel('Eigenvalues');
plt.show()

# Tolerance epsilon to determine the number of modes Nrb
print('Give a tolerance to compute Nrb')
epsilon_POD = float(input())

# Computation of the number of modes Nrb
Nrb = energy_number(epsilon_POD,Eigen_val_normalized)
print('This corresponds to Nrb = ',Nrb)
ans = input('type any key to resume')
            
# Truncation of the Reduced Basis 
Brb = Xi[:,0:Nrb]
print("Brb",Brb)
t_1 =  time.time()
# The error estimation satisfied by the POD method
Uh = np.zeros(NN)
sum = 0.
#Error_matrix = np.zeros((M,NN))
for m in range(M):
    # The FE solution
    Uh = Usnap[:,m]
    # The L2 projection
    Urb = np.dot(Brb,np.dot(Brb.transpose(),Uh))
    # difference function Uh-Urb
    diff = Uh - Urb
    sum = sum + np.dot(diff.transpose(),diff)

discarded_eigenvalues =  np.sum(Eigen_val[Nrb:]) # useful to validate the error estimation, see later

