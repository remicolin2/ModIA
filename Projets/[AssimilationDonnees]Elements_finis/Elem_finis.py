import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from fenics import *
import dolfin as fe
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt 
#
# FUNCTIONS
#

# transform a npy array to a FEniCS field
def numpy_to_fenics(array, function_space):
	function = Function(function_space)
	for i in range(len(function.vector()[:])):
		function.vector()[i] = array[i]
	return function

###############
###### Définition du domaine 
NP = 30;
print('Number of mesh points NP = ', NP)
mesh = UnitSquareMesh(NP, NP)

class Wall(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            (fe.near(x[0], 1)) ## mur de droite
            or (fe.near(x[0], 0)) ## mur de gauche 
            or (fe.near(x[1], 0)) ##plafond
        )
    
class Gamma_Fire(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            (fe.near(x[1], 0)  and (x[0] >= 0.5 and x[0]<= 1))
        )

class Gamma_Dir(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            (fe.near(x[1], 0)  and (x[0] >= 0 and x[0]<= 0.5))
        )

# create a cell function over the boundaries edges
sub_boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1) # mesh.topology().dim()-1
sub_boundaries.set_all(0)

wall = Wall()
wall.mark(sub_boundaries, 1)

fire = Gamma_Fire()
fire.mark(sub_boundaries, 2)

gamma_dir = Gamma_Dir()
gamma_dir.mark(sub_boundaries, 3)

domains = fe.MeshFunction("size_t", mesh, mesh.topology().dim())  # CellFunction
domains.set_all(0)

# redefining integrals over boundaries
ds = fe.Measure('ds', domain=mesh, subdomain_data=sub_boundaries)

# Define new measures associated with the interior domains
dx = fe.Measure("dx", domain=mesh, subdomain_data=domains)

plot(mesh, title='Mesh')
plt.show(block=True)
############################################################


### Définir les constantes 
fire_temp = fe.Constant(1273.15) # fire at 1000 °C
sigma = fe.Constant(5.64e-8)
mu = fe.Constant(1.9e-5)
wall_temp = fe.Constant(273.15) # Wall at 0°C
c = fe.Constant(0)
f = fe.Constant(0)

###############################
os.system("clear") 
print('#')
print('# MAIN PROGRAM')
print('#')

###################################################################
# Solutionx stationnaire
###################################################################


################################
#### Construction des P-2 Lagrande et définition de u et v
k = 2 ; print('Order of the Lagrange FE k = ', k)
V = fe.FunctionSpace(mesh, "CG", int(k)) # Lagrange FE, order k
# Fonctions trial et test 
u = fe.TrialFunction(V)
v = fe.TestFunction(V)

################################
###### Conditions de dirichlet 
u_diri_homog = Expression('0.', degree=u.ufl_element().degree())             
u_diri_fire = fe.DirichletBC(V, u_diri_homog, fire)
u_diri_wall = fe.DirichletBC(V, wall_temp, wall)
u_gamma_dir  = fe.DirichletBC(V, u_diri_homog, gamma_dir)

#############################
#### Ordre de la solution 
omega_1 = 2.0
omega_2 = 3.0
u_ex = Expression("cos(omega_1 * x[0]) * sin(omega_2 * x[1])", degree=2, omega_1=omega_1, omega_2=omega_2)

#Calcul du terme source :
#f_bulk = -mu * div(grad(u_ex))  

print('##################################################################')
print('#')
print('# Algorithme de Newton-Raphson')
print('#')
print('##################################################################')

print('#')
print('# Initialization: u0 solution of a semi-linearized BVP')
print('#')

print('#')
print('# Iterations')
print('#') 
i_max = 100 # max of iterations
eps_du = 1e-9 # tolerance on the relative norm

du = fe.TrialFunction(V)
un, dun = fe.Function(V), fe.Function(V)

u_init = fe.Expression("293.15", degree=1) # Room at 20°C at time t=0
u = fe.interpolate(u_init, V)
un = u 
nb_pas = 10
iteration_numbers = []
errors = []


##################
# Loop
i = 0
error = eps_du+1 # current iteration
while (error>eps_du and i<i_max): 
    i+=1 # update the current iteration number
    print("Newton-Raphson iteration #",i," begins...")
    # LHS of the linearized variational formulation
    a = mu*fe.inner(grad(du), grad(v))*dx-sigma*4*un**3*du*v*ds(2)
    # RHS of the linearized eqn
    L = - mu*fe.inner(grad(un), grad(v))*dx + sigma*(un**4-fire_temp**4)*v*ds(2)
    # Solve
    fe.solve(a == L, dun, u_gamma_dir)
    error = np.linalg.norm((dun.vector().get_local())) / np.linalg.norm(un.vector().get_local())
    iteration_numbers.append(i)
    errors.append(error)
    un.assign(un+dun) # update the solution  
    print("Newton-Raphson iteration #",i,"; error = ", error)
    # test
    if (i == i_max):
        print("Warning: the algo exits because of the max number of ite ! error = ",error)
    
if (i < i_max):
  print("* Newton-Raphson algorithm has converged: the expected stationarity has been reached. eps_du = ",eps_du)
#
# Plots
#
fe.plot(mesh)
p=fe.plot(un, title='Stationary non-linear solution ')
p.set_cmap("rainbow"); plt.colorbar(p); plt.show()
plt.show()


##### 
### Calcul de l'erreur entre la solution exacte et la solution numérique 
error_norm = errornorm(u_ex, un, 'L2')

plt.figure()
plt.loglog(iteration_numbers, errors, marker='o', linestyle='-', color='b', label='Computed Error')
plt.axhline(y=error_norm, color='r', linestyle='--', label='True Error')
plt.title('Convergence Plot')
plt.xlabel('Iteration Number')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()  
plt.show()

###################################################################
# Solution non stationnaire
###################################################################

##############################
## Construction des P-2 Lagrande et définition de u et v
k = 2 ; print('Order of the Lagrange FE k = ', k)
V = fe.FunctionSpace(mesh, "CG", int(k)) # Lagrange FE, order k
# Fonctions trial et test 
u = fe.TrialFunction(V)
v = fe.TestFunction(V)

##############################
#### Conditions de dirichlet              
u_diri_homog = Expression('0.', degree=u.ufl_element().degree())             
u_diri_fire = fe.DirichletBC(V, u_diri_homog, fire)
u_diri_wall = fe.DirichletBC(V, wall_temp, wall)
u_gamma_dir  = fe.DirichletBC(V, u_diri_homog, gamma_dir)
#############################
### Initialisation du temps 
t1 = 0.0
t2 = 1e5
n = 1000
dt = (t2 - t1) / n
t = t1

###########################
## Ordre de la solution 
omega_1 = 2.0
omega_2 = 3.0
u_ex = Expression("cos(omega_1 * x[0]) * sin(omega_2 * x[1])", degree=2, omega_1=omega_1, omega_2=omega_2)

# Calcul du terme source :
# f_bulk = -mu * div(grad(u_ex))  

print('##################################################################')
print('#')
print('# Algorithme de Newton-Raphson')
print('#')
print('##################################################################')

print('#')
print('# Initialization: u0 solution of a semi-linearized BVP')
print('#')

print('#')
print('# Iterations')
print('#') 
i_max = 100 # max of iterations
eps_du = 1e-9 # tolerance on the relative norm

du = fe.TrialFunction(V)
un, un_plus_1, dun = fe.Function(V), fe.Function(V), fe.Function(V)

u_init = fe.Expression("293.0", degree=1)
u = fe.interpolate(u_init, V)
un.assign(u)
un_plus_1.assign(u)
nb_pas = 200
iteration_numbers = []
errors = []

errors_time = []
# Loop
for t_ in range(nb_pas):
    un_plus_1.assign(un)
    i = 0
    error = eps_du+1 # current iteration
    while (error>eps_du and i<i_max): 
        i+=1 # update the current iteration number
        print("Newton-Raphson iteration #",i," begins...")
        # LHS of the linearized variational formulation
        a = du * v * dx + dt * mu * inner(grad(du), grad(v)) * dx - dt * sigma * 4 * un_plus_1**3 * du * v * ds(2)
        # RHS of the linearized eq
        L = - dt * sigma * fire_temp**4 * v * ds(2) + un * v * dx - un_plus_1 * v * dx -dt * mu * inner(grad(un_plus_1), grad(v)) * dx+ dt * sigma * un_plus_1**4 * v * ds(2)
        # Solve
        fe.solve(a == L, dun, u_gamma_dir)
        error = np.linalg.norm((dun.vector().get_local())) / np.linalg.norm(un_plus_1.vector().get_local())
        un_plus_1.assign(un_plus_1+dun)
        print("Newton-Raphson iteration #",i,"; error = ", error)
        # test
        if (i == i_max):
            print("Warning: the algo exits because of the max number of ite ! error = ",error)
    un.assign(un_plus_1) # update the solution

    # error calculation at this time 
    current_error = errornorm(u_ex, un_plus_1, 'L2')
    errors_time.append(current_error)
    print(f"Error at time {t}: {current_error}")
    
    if (i < i_max):
        print("* Newton-Raphson algorithm has converged: the expected stationarity has been reached. eps_du = ",eps_du)
# 
# Plots
# 
fe.plot(mesh)
p=fe.plot(un, title=f'Non-linear solution at time t = {t2}')
p.set_cmap("rainbow"); plt.colorbar(p); plt.show()
plt.show()


### 
# Error calculation
plt.figure()
plt.loglog(np.linspace(1, nb_pas, nb_pas), errors_time, marker='o', linestyle='-', color='b', label='Computed Error')
plt.axhline(y=error_norm, color='r', linestyle='--', label='True Error')
plt.title('Convergence Plot over Time')
plt.xlabel('Time Step')
plt.ylabel('Error (log)')
plt.legend()
plt.grid(True)
plt.show()