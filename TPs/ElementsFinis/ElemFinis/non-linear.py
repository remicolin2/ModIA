'''
NON linear BVP:
     - div( mu(u) * grad(u) ) + w * grad(u) = f  in domain
                                           u = g  on bdry dirichlet
                         - mu(u) nabla(u).n = 0 on bdry Neumann
    with w: given velocity field; mu: given diffusivity coeff. 
    
Example of basic exact solution in domain=(0,1)^2: 
        u = 'sin(x[0]) + sin(x[1])' corresponds to: 
        f = 'cos(x[0]) + cos(x[1]) + sin(x[0]) + sin(x[1])' and g = 'sin(x[0]) + sin(x[1])'
'''

# from dolfin import *
import sys
from fenics import *
from sys import exit
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


#
# FUNCTIONS
#

# transform a npy array to a FEniCS field
def numpy_to_fenics(array, function_space):
    function = Function(function_space)
    for i in range(len(function.vector()[:])):
        function.vector()[i] = array[i]
    return function


#
# Dirichlet boundary conditions
#
# The functions below return True for points inside the subdomain and False for the points outside.
# Because of rounding-off errors, we specify |𝑥−1|<𝜖, where 𝜖 is a small number (such as machine precision).
tol_bc = 1e-7


def u_bdry_x0(x, on_boundary):  # Left bdry
    return bool(on_boundary and (near(x[0], 0, tol_bc)))


def u_bdry_x1(x, on_boundary):  # Right bdry
    return bool(on_boundary and (near(x[0], 1., tol_bc)))


#
# The non linear parameter m(u) and its derivative
#
m = 5  # power-law index
# print('The power-law exponent of the non linearity m = ', m)
print('The expression of mu(u) is ... see function !')


def mu(u):
    return (0.1 + u) ** int(m)  # non linear law
    # return (1. + 1.e-30*u) # ugly way to consider an almost linear equation
    # return 1.


def dmu_du(u):
    return m * (0.1 + u) ** int(m - 1)
    # return 1.e-30


# return u * 0.

###############################
# os.system("clear")
print('#')
print('# MAIN PROGRAM')
print('#')

# Create mesh and function space
NP = 30;
print('Number of mesh points NP = ', NP)
mesh = UnitSquareMesh(NP, NP)
k = 2;
print('Order of the Lagrange FE k = ', k)
V = FunctionSpace(mesh, "CG", int(k))  # Lagrange FE, order k

advection = False
if advection:
    # Define velocity field
    # Be creative ! Define your own velocity field following the instructions
    vel_amp = 1.e+2  # ; print('vel_amp =',vel_amp)
    vel_exp = Expression(('(1.+abs(cos(2*pi*x[0])))', 'sin(2*pi/0.2*x[0])'), element=V.ufl_element())
    # vel_exp = Expression(('(1.+abs(cos(2*pi*x[0])))', 'sin(2*pi/0.2*x[0])'), element = V.ufl_element())
    # vel_exp = Expression(('0.', '0.'), element = V.ufl_element())
    V_vec = VectorFunctionSpace(mesh, "CG", k)
    vel = vel_amp * interpolate(vel_exp, V_vec)
    # plot
    p = plot(vel, title='The velocity field')
    p.set_cmap("rainbow");
    plt.colorbar(p);
    plt.show()

# To transform a vector vec to a fenics object vf
# vf= Function(V); vf.vector().set_local(vec)

# The physical RHS 
# f_exp = Expression('1.', element = V.ufl_element())
fp_exp = Expression('1e+15 * exp( -( abs(x[0]-0.5) + abs(x[1]-0.5) ) / 0.1 )', element=V.ufl_element())
fp = interpolate(fp_exp, V)
# fp = Expression('0.', degree=u.ufl_element().degree())
plt.figure()
p2 = plot(fp, title='The source term')
p2.set_cmap("rainbow");
plt.colorbar(p2);
plt.show()  # (block=True)

print('##################################################################')
print('#')
print('# Newton - Raphson algorithm: Home-implemented non linear solver :)')
print('#')
print('##################################################################')

print('#')
print('# Initialization: u0 solution of a semi-linearized BVP')
print('#')

# Trial & Test functions
u = TrialFunction(V);
v = TestFunction(V)

#
# Boundary conditions
# Dirichlet b.c.
u_diri_non_homog = Expression('293.', degree=u.ufl_element().degree())
u_diri_homog = Expression('0.', degree=u.ufl_element().degree())
bc = DirichletBC(V, u_diri_non_homog, u_bdry_x0)

# Diffusivity coeff. depending on the field u0
u0_mu_exp = u_diri_non_homog
mu0 = mu(interpolate(u0_mu_exp, V))

# A semi-linearized pb
if advection:
    F0 = dot(mu0 * grad(u), grad(v)) * dx + dot(vel, grad(u)) * v * dx - fp * v * dx
    # Add the SUPG stabilisation terms
    vnorm = sqrt(dot(vel, vel))
    h = MaxCellEdgeLength(mesh)
    delta = h / (2.0 * vnorm)
    residual = - div(mu0 * grad(u)) + dot(vel, grad(u)) - fp  # the residual expression
    F0 += delta * residual * dot(vel, grad(v)) * dx  # the enriched weak formulation
else:
    F0 = dot(mu0 * grad(u), grad(v)) * dx - fp * v * dx
# The bilinear and linear forms
a0 = lhs(F0);
L0 = rhs(F0)

# Neumann bc
# Nothing to do since they are here homogeneous !
# F0 += int-de-bord

# Solve the linear system
u0 = Function(V)
solve(a0 == L0, u0, bc)  # , [bc0,bc1])

if advection:
    # Peclet number(s)
    Pe = 0.5 * sqrt(dot(vel, vel)) / mu0
    Pe_np = project(Pe, V).vector().get_local()
    hmax = mesh.hmax();  # print(type(hmax))
    Peh_np = hmax * Pe_np
    print('Peclet number Pe: approx. (min., max.) values for u0 : ', '{0:.2f}'.format(Pe_np.min()),
          '{0:.2f}'.format(Pe_np.max()))
    print('Num. Peclet number Pe_h=(h.Pe): approx. (min., max.) values for u0 : ', '{0:.2f}'.format(Peh_np.min()),
          '{0:.2f}'.format(Peh_np.max()))
    ans = print("   press any key to resume...")

# Plot the solution
plt.figure()
plot(mesh)
p = plot(u0, title='The built up initial solution u0')
p.set_cmap("rainbow");
plt.colorbar(p);
plt.show(block=False)

print('#')
print('# Iterations')
print('#')
i_max = 40  # max of iterations
i = 0;
error = 1.  # current iteration
eps_du = 1e-9  # tolerance on the relative norm

# The FE unknowns 
du = TrialFunction(V)
un, dun = Function(V), Function(V)
un = u0  # initialisation

f = Constant(-6.0)
# Loop
while (error > eps_du and i < i_max):
    i += 1  # update the current iteration number
    print("Newton-Raphson iteration #", i, " begins...")

    # mu and dmu_du at the current iteration
    mu_n = mu(un)
    dmu_du_n = dmu_du(un)

    # LHS of the linearized variational formulation
    a = (- mu_n * dot(grad(du), grad(v)) + dmu_du_n * du * dot(grad(un), grad(v))) * dx  # TO BE COMPLETED
    # RHS of the linearized eqn

    L = (- mu_n * dot(grad(un), grad(v)) + f * v) * dx  # TO BE COMPLETED

    # Homogeneous Dirichlet b.c.
    bc0 = DirichletBC(V, u_diri_homog, u_bdry_x0)

    # Solve
    solve(a == L, dun, bc0)
    un.assign(un + dun)  # update the solution

    # relative diff.

    error = np.linalg.norm(du) / np.linalg.norm(un)

    print("Newton-Raphson iteration #", i, "; error = ", error)
    # test
    if (i == i_max):
        print("Warning: the algo exits because of the max number of ite ! error = ", error)

if (i < i_max):
    print("* Newton-Raphson algorithm has converged: the expected stationarity has been reached. eps_du = ", eps_du)
#
# Plots
#
plt.figure();
plot(mesh)
p = plot(un, title='The non linear solution (home-made solver)')
p.set_cmap("rainbow");
plt.colorbar(p);
plt.show(block=False)
