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


question5=False # Si = True : créé le plot des erreurs de la question 5 
question7=False # Si = True, change la matrice B pour la question 7

#  Discrétisation en espace
xmin = 0.0; xmax = 1.0; nptx = 16; nx = nptx-2  ; mu=1
hx = (xmax-xmin)/(nptx -1)
xx = np.linspace(xmin,xmax,nptx) 
xx = xx.transpose()
xxint = xx[1:nx+1]
ymin = 0.0; ymax = 1.0; npty = 16; ny = npty-2 
hy = (ymax-ymin)/(npty -1)
yy = np.linspace(ymin,ymax,npty)
yy=yy.transpose() 
yyint = yy[1:ny+1]

"""
#  Matrix system
# On Ox
Kx # Local matrix of size Nx+2 relative to Ox discretization
K2Dx # Global Matrix of (Ny+2)**2 matrices of size (Nx+2)**2

# On Oy
Ky # Local matrix of size Nx+2 relative to Oy discretization
K2Dy # Global Matrix of (Ny+2)**2 matrices of size (Nx+2)**2
#
#
K2D = K2Dx + K2Dy # Final matrix of Laplacien operator with Dirichlet Boundary conditions
"""

##  Solution and source terms
u = np.zeros((nx+2)*(ny+2)) #Numerical solution
u_ex = np.zeros((nx+2)*(ny+2)) #Exact solution
F = np.zeros((nx+2)*(ny+2)) #Source term
#
#
# Source term
def Source_int(x):
    return 2*np.pi**2*(np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
def Source_bnd(x):
    return np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
def Sol_sin(x):
    return np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
for i in range(nptx):
    for j in range(npty):
        coord = np.array([i*hx,j*hy])
        u_ex[j*(nx+2) + i] = Sol_sin(coord)
    if i==0 or i==nptx-1: # Boundary x=0 ou x=xmax
        for j in range(npty):
            coord = np.array([i*hx,j*hy])
            F[j*(nx+2) + i]=Source_bnd(coord)
    else:
        for j in range(npty):
            coord = np.array([i*hx,j*hy])
            if j==0 or j==npty-1: # Boundary y=0 ou y=ymax
                F[j*(nx+2) + i]=Source_bnd(coord)
            else:
                F[j*(nx+2) + i]=Source_int(coord)



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
        B_mat[0, 0] = B_mat[-1, -1] = 1
        B_mat[0, 1] = B_mat[0, -1] = 0
        B_mat[-1, -2] = B_mat[-1, 0] = 0

        if question7==True:
            B_mat[-1, -1] = -3/(2*hx)
            B_mat[-1, -2] = 2/hx
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



if __name__== '__main__':  
    #Post-traitement u_ex+Visualization of the exct solution
    uu_ex = np.reshape(u_ex,(nx+2 ,ny+2),order = 'F');
    X,Y = np.meshgrid(xx,yy)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y,uu_ex.T,rstride = 1, cstride = 1,cmap="coolwarm");

    
    #        
    K2D=K2Dx(nx,ny)
    u = npl.solve(K2D,F)
    uu = np.reshape(u,(nx+2 ,ny+2),order = 'F');
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y,uu.T,rstride = 1, cstride = 1,cmap="coolwarm");
    plt.show()

    # Error
    print('norm L2 = ',npl.norm(u-u_ex))

if question5==True:
    erreur=[]
    for npt in range(5,55,5):
        #  Discrétisation en espace
        xmin = 0.0; xmax = 1.0; nptx = npt; nx = nptx-2  ; mu=1
        hx = (xmax-xmin)/(nptx -1)
        xx = np.linspace(xmin,xmax,nptx) 
        xx = xx.transpose()
        xxint = xx[1:nx+1]
        ymin = 0.0; ymax = 1.0; npty = npt; ny = npty-2 
        hy = (ymax-ymin)/(npty -1)
        yy = np.linspace(ymin,ymax,npty)
        yy=yy.transpose() 
        yyint = yy[1:ny+1]

        """
        #  Matrix system
        # On Ox
        Kx # Local matrix of size Nx+2 relative to Ox discretization
        K2Dx # Global Matrix of (Ny+2)**2 matrices of size (Nx+2)**2

        # On Oy
        Ky # Local matrix of size Nx+2 relative to Oy discretization
        K2Dy # Global Matrix of (Ny+2)**2 matrices of size (Nx+2)**2
        #
        #
        K2D = K2Dx + K2Dy # Final matrix of Laplacien operator with Dirichlet Boundary conditions
        """

        ##  Solution and source terms
        u = np.zeros((nx+2)*(ny+2)) #Numerical solution
        u_ex = np.zeros((nx+2)*(ny+2)) #Exact solution
        F = np.zeros((nx+2)*(ny+2)) #Source term
        #
        #
        # Source term
        def Source_int(x):
            return 2*np.pi**2*(np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
        def Source_bnd(x):
            return np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
        def Sol_sin(x):
            return np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
        for i in range(nptx):
            for j in range(npty):
                coord = np.array([i*hx,j*hy])
                u_ex[j*(nx+2) + i] = Sol_sin(coord)
            if i==0 or i==nptx-1: # Boundary x=0 ou x=xmax
                for j in range(npty):
                    coord = np.array([i*hx,j*hy])
                    F[j*(nx+2) + i]=Source_bnd(coord)
            else:
                for j in range(npty):
                    coord = np.array([i*hx,j*hy])
                    if j==0 or j==npty-1: # Boundary y=0 ou y=ymax
                        F[j*(nx+2) + i]=Source_bnd(coord)
                    else:
                        F[j*(nx+2) + i]=Source_int(coord)



        #MATRICE K2Dx
        def K2Dx(Nx, Ny):

            #MATRICE Kx
            def Kx(Nx):
                A_mat = (-mu/hy**2)*np.eye((Nx+2))
                A_mat[0, 0] = A_mat[-1, -1] = 0
                return A_mat

            #MATRICE Ky
            def Ky(Nx):
                B_mat = mu*((2/hx**2)+(2/hy**2))*np.eye((Nx+2)) + (-mu/hx**2)*np.roll(np.eye((Nx+2)), shift=1, axis=1) + (-mu/hx**2)*np.roll(np.eye((Nx+2)), shift=-1, axis=1)
                B_mat[0, 0] = B_mat[-1, -1] = 1
                B_mat[0, 1] = B_mat[0, -1] = 0
                B_mat[-1, -2] = B_mat[-1, 0] = 0
                return B_mat

            kronKx = np.roll(np.eye(Ny+2), shift=+1, axis=1) + np.roll(np.eye(Ny+2), shift=-1, axis=1)
            kronKx[0, -1] = kronKx[-1, 0] = 0
            kronKx[0, 1] = kronKx[-1, -2] = 0
            A_mat = Kx(Nx)

            kronKy = np.eye(Ny+2)
            kronKy[0, 0] = kronKy[-1, -1] = 0
            B_mat = Ky(Nx)

            kronI = np.zeros((Ny+2, Ny+2))
            kronI[0, 0] = kronI[-1, -1] = 1
            Im = np.eye(Nx+2)

            K2Dx_mat = np.kron(kronI, Im) + np.kron(kronKx, A_mat) + np.kron(kronKy, B_mat)
            return K2Dx_mat



    

        #        
        K2D=K2Dx(nx,ny)
        u = npl.solve(K2D,F)

        erreur.append(npl.norm(u-u_ex))

    plt.plot([i for i in range(5,55,5)],erreur)
    plt.xlabel("Nombre de points")
    plt.ylabel("Erreur")
    plt.title("Erreur en fonction du nombre de points")
    plt.show()
