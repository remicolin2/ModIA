#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 05/02/2023

@author: Rémi Colin, Mickaël Song
"""

import numpy as np
import numpy.linalg as npl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

nicolson=True # Si nicolson = True, on utilise la méthode de Cranck-Nicholson, 
# sinon on utilise la méthode d'Euler explicite
question13=True # A mettre = True pour changer la CI pour vérifier la question 13


#  Discrétisation en espace
xmin = 0.0; xmax = 2; nptx = 61; nx = nptx-2  
hx = (xmax-xmin)/(nptx -1)
xx = np.linspace(xmin,xmax,nptx) 
xx = xx.transpose()
xxint = xx[1:nx+1]
ymin = 0.0; ymax = 1.0; npty = 61; ny = npty-2 
hy = (ymax-ymin)/(npty -1)
yy = np.linspace(ymin,ymax,npty)
yy=yy.transpose() 
yyint = yy[1:ny+1]

# =============================================================================
### Parameters
mu = 0.01 # Diffusion parameter
vx = 1 # Vitesse along x
# =============================================================================

cfl = 0.2  # cfl =mu*dt/hx^2+mu*dt/hy^2 ou v*dt/h
dt = (hx**2)*(hy**2)*cfl/(mu*(hx**2 + hy**2)) # dt = pas de temps
dt = cfl*hx/vx
Tfinal = 0.8   # Temps final souhaité



###### Matrice de Diffusion Dir/Neumann
#MATRICE K2Dx
def K2Dx(Nx, Ny):

    #MATRICE A
    def A(Nx):
        A_mat = (-mu/hy**2)*np.eye((Nx+2))
        A_mat[0, 0] = A_mat[-1, -1] = 0
        return A_mat

    #MATRICE B
    def B(Nx):
        B_mat = mu*((2/hx**2)+(2/hy**2))*np.eye((Nx+2)) + (-mu/hx**2)*np.roll(np.eye((Nx+2)), shift=1, axis=1) + (-mu/hx**2)*np.roll(np.eye((Nx+2)), shift=-1, axis=1)
        B_mat[0, 0] = 1
        B_mat[0, 1] = B_mat[0, -1] = 0
        B_mat[-1, -2] = B_mat[-1, 0] = 0
        B_mat[-1, -1] = -3/(2*hx)
        B_mat[-1, -2] = 4/(2*hx)
        B_mat[-1, -3] = -1/(2*hx)
        return B_mat

    kronKx = np.roll(np.eye(Ny+2), shift=+1, axis=1) + np.roll(np.eye(Ny+2), shift=-1, axis=1)
    kronKx[0, -1] = kronKx[-1, 0] = 0
    kronKx[0, 1] = kronKx[-1, -2] = 0
    A_mat = A(Nx)

    kronKy = np.eye(Ny+2)
    kronKy[0, 0] = kronKy[-1, -1] = 0
    B_mat = B(Nx)

    kronI = np.zeros((Ny+2, Ny+2))
    kronI[0, 0] = kronI[-1, -1] = 1
    Im = np.eye(Nx+2)

    K2Dx_mat = np.kron(kronI, Im) + np.kron(kronKx, A_mat) + np.kron(kronKy, B_mat)
    return K2Dx_mat



#### Matrice de Convection  (Centré)
def Conv(Nx, Ny):
  def V(Nx):
    V_mat = vx*(np.diag([0] + [0.5/hx]*Nx, 1) - np.diag([0.5/hx]*Nx + [0], -1))
    return V_mat

  kronV = np.eye(Ny+2)
  kronV[0, 0] = kronV[-1, -1] = 0
  V_mat = V(Nx)
  Conv_mat = np.kron(kronV, V_mat)
  return Conv_mat


#### Global matrix : diffusion + convection
K2D = K2Dx(nx, ny)

V2Dx = Conv(nx, ny)

A2D = (K2D + V2Dx) #-mu*Delta + V.grad
#
#
##  Cas explicite
u = np.zeros((nx+2)*(ny+2))
u_ex = np.zeros((nx+2)*(ny+2))
err = np.zeros((nx+2)*(ny+2))
F = np.zeros((nx+2)*(ny+2))
#
#
# =============================================================================
# Time stepping
# =============================================================================
s0 = 0.1
x0 = 0.25
y0=0.5

def Sol_init(x):
    return np.exp( -((x[0]-x0)/s0)**2 -((x[1]-y0)/s0)**2   )

def Sol_init_question13(x):
  return np.maximum(0, s0**2 - (x[0] - x0)**2 - (x[1] - y0)**2)





u_init = np.zeros((nx+2)*(ny+2))
for i in range(nptx):
     for j in range(npty):
      coord = np.array([xmin+i*hx,ymin+j*hy])
      if question13==False:
        u_init[j*(nx+2) + i] = Sol_init(coord)
      else:
        u_init[j*(nx+2) + i] = Sol_init_question13(coord)

             
uu_init = np.reshape(u_init,(nx+2 ,ny+2),order = 'F');
fig = plt.figure(figsize=(10, 7))
X,Y = np.meshgrid(xx,yy)
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X, Y, uu_init.T, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
ax.view_init(60, 35)
plt.pause(1.)
             
## Initialize u by the initial data u0
u = u_init.copy()

# Nombre de pas de temps effectues
nt = int(Tfinal/dt)
Tfinal = nt*dt # on corrige le temps final (si Tfinal/dt n'est pas entier)

concentration=[]
# Time loop
for n in range(1,nt+1):

  # Schéma explicite en temps
  if nicolson==True:
    #Schéma de Crank-Nicolson
    I = np.eye((nx+2)*(ny+2))
    u = np.linalg.solve(I+0.5*dt*A2D, u + 0.5*dt*np.dot(-A2D, u))
  else:
    # Schéma d'Euler explicite
    u = u - dt*np.dot(A2D, u)

 # Print solution
  if n%5 == 0:
      plt.figure(1)
      plt.clf()
      fig = plt.figure(figsize=(10, 7))
      ax = plt.axes(projection='3d')
      uu = np.reshape(u,(nx+2 ,ny+2),order = 'F');
      surf = ax.plot_surface(X, Y, uu.T, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
      ax.view_init(60, 35)
      plt.title(['Schema explicite avec CFL=%s' %(cfl), '$t=$%s' %(n*dt)])
      plt.pause(0.1)

      concentration.append(np.max(u))

plt.close("all")
# Question 12/13
plt.plot([dt*i*5 for i in range(len(concentration))],concentration)
plt.xlabel("temps (s)")
plt.ylabel("Concentration")
plt.title("Concentration en fonction du temps")
plt.show()


####################################################################
# comparaison solution exacte avec solution numerique au temps final
j0 = int((npty-1)/2)

plt.figure(2)
plt.clf()
x = np.linspace(xmin,xmax,nptx)
plt.plot(x,uu_init[:,j0],x,uu[:,j0],'k') #,x,uexacte,'or')
plt.legend(['Solution initiale','Schema explicite =%s' %(cfl)]) #,'solution exacte'],loc='best')
plt.show()
