import fenics as fe
import matplotlib.pyplot as plt


FORCING_MAGNITUDE = 1.0
NB_POINTS_PAR_AXES = 12

mesh = fe.UnitSquareMesh(NB_POINTS_PAR_AXES, NB_POINTS_PAR_AXES)
lagrange_polynomial_space_first_order = fe.FunctionSpace(mesh, "CG", 1)

def bundary_boolean_function(x, on_boundary):
    return on_boundary

constant_value = 0.0 

homogeneous_dirichlet_boundary_condition = fe.DirichletBC(
    lagrange_polynomial_space_first_order,
    fe.Constant(constant_value, cell=mesh.ufl_cell()),
    bundary_boolean_function,
)

# trial and test funcitons
u_trial = fe.TrialFunction(lagrange_polynomial_space_first_order)
v_test = fe.TrialFunction(lagrange_polynomial_space_first_order)

#Weak form 
forcing = fe.Constant(-FORCING_MAGNITUDE, cell=mesh.ufl_cell())
weak_form_left = fe.dot(fe.grad(u_trial), fe.grad(v_test)) * fe.dx
weak_form_right = forcing * v_test* fe.dx

# Finite element assembly and linear solution
u_solution = fe.Function(lagrange_polynomial_space_first_order)
fe.solve(
    weak_form_left == weak_form_right,
    u_solution,
    homogeneous_dirichlet_boundary_condition,
)

c = fe.plot(u_solution, mode = "color")
plt.colorbar(c)
fe.plot(mesh)
plt.show()