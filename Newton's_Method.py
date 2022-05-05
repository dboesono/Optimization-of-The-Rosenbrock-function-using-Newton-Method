from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
from mpl_toolkits import mplot3d
from matplotlib import cm
from sympy import *
import numpy
from sympy.abc import x, y
from sympy import ordered, Matrix, hessian
from math import *
from matplotlib.ticker import MaxNLocator
from itertools import product
import autograd.numpy as np
from autograd import grad, jacobian
import sympy as smp
from sympy.abc import x,y
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.pyplot import *
from numpy import *
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def objective(x):
    # The objective(x) function returns the rosenbrock function with a = 1 and b = 100
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


def gradient(x):
    # The gradient function computes the partial derivative of each variable and store it into a array.
    grad = list()
    dfdx = 2*(x[0] - 1) - 4*100*x[0]*(x[1] - x[0]**2)
    dfdy = 2 * 100 * (x[1] - x[0]**2)
    grad.append(dfdx)
    grad.append(dfdy)
    return np.array(grad)


def hessian(x):
    # The hessian function computes the second partial derivative w.r.t to every variable and
    # returns its values in matrix form
    dfdx2 = 2 - 4*100*(x[1] - 3*x[0]**2)
    dfdxdy = -4 * 100 * x[0]
    dfdydx = -4 * 100 * x[0]
    dfdy2 = 2 * 100
    return np.array([[dfdx2, dfdxdy], [dfdydx, dfdy2]])


def l2_norm(x):
    # The l2_norm function computes the L2 norm of a 2 dimension vector.
    return (x[0]**2 + x[1]**2)**0.5


def backtracking_line_search(gamma, sigma, xk, dk):
    """
    The backtracking function Minimize over the alpha from the function f(xk + alpha*dk) and alpha > 0 is assumed to
    be a descent direction

    Parameters
    ----------
    xk: A current point from the function that takes the form of an array.

    dk: The search or descent direction usually takes the value of the first
        derivative array of the function.

    sigma: The value of the alpha shrinkage factor that takes the form of a float.

    gamma: The value to control the stopping criterion that takes the form of a float.

    Returns
    -------
    alpha: The value of alpha at the end of the optimization that takes the form of a scalar.

    """
    # Initialize the alpha equal to the value 1
    alpha = 1

    # Began the procedure based on the Backtracking/Armijo line search condition in the while loop
    while True:
        if objective(xk + alpha * dk) <= objective(xk) + gamma * alpha * np.inner(dk, gradient(xk)):
            return alpha
        else:
            alpha = alpha * sigma


def newton_method(obj, grad, hess, init=[-1, -0.5], tol=10 ** (-7), gamma=10 ** (-4), sigma=0.5, gamma1=10 ** (-6),
                  gamma2=0.1):
    """
    Newton method for unconstrained optimization problem given a starting point x which is a element
    of real numbers. The algorithm will repeat itself according to the following procedure:

    1. Define the descending direction newton method condition.
    2. Using a step size strategy, choose the step length alpha using the Armijo Line Search/Backtracking strategy.
    3. Update the x point using the formula of x := x + alpha*direction

    Repeat this procedure until a stopping criterion is satisfied.

    Parameters
    ----------
    obj: The objective function f(x)

    grad: The gradient vector of the objective function f(x)

    hess: The hessian matrix of the objective function f(x)

    init: The initial value of x and y in the form of an array

    tol: The tolerance for the l2 norm of f_grad

    gamma1: parameter for checking whether a direction is a good descent direction

    gamma2: parameter for checking whether a direction is a good descent direction

    gamma: The value to control the stopping criterion that takes the form of a float

    sigma: The value of the alpha shrinkage factor that takes the form of a float.

    Returns
    -------
    Solutions: The vector of the coordinates in the learning path.

    values: The value of the objective function along the learning path.

    """

    # Initialize the initial point x0 as an array and create two arrays to store the x and f(x) values
    # Initialize the number of iterations
    xk = np.array(init)
    curve_x = [xk]
    curve_y = [obj(xk)]
    num_iter = 0
    print('Initial Condition: f(x) = {}, x = {} \n'.format(obj(xk), xk))

    # Utilize the stopping criterion when the l2 norm of the objective function is less than the tolerance
    # value to break the output out of the while loop
    while l2_norm(grad(xk)) > tol:

        # Compute the descent direction of the newton method algorithm
        # First compute sk by solving the system of equation of hessian(x) * sk = -(gradient(x))
        sk = np.linalg.solve(hess(xk), -1 * grad(xk))

        # Next decide whether sk is a good descent direction based on the condition of newton method
        # The condition is -gradient(x) * sk >= gamma1 * min{1, ||sk||**gamma2} * ||sk||**2
        # If the sk satisfy the condition then accept the newton direction and set dk = sk
        # Otherwise, set dk = -(gradient(x))
        if np.inner(-1 * grad(xk), sk) >= gamma1 * min(1, l2_norm(sk) ** gamma2) * (l2_norm(sk) ** 2):
            dk = sk
            print(f"Iteration {num_iter + 1} Direction: sk")
        else:
            dk = -1 * grad(xk)
            print(f"Iteration {num_iter + 1} Direction: dk")

        # Compute the alpha step size using backtracking/armijo line search strategy
        alpha = backtracking_line_search(gamma, sigma, xk, dk)

        # Update the x values using the equation x^k+1 = x^k + alpha * dk
        xk = xk + alpha * dk

        # Store values into the curve_x and curve_y arrays
        curve_x.append(xk)
        curve_y.append(obj(xk))

        # Update number of iterations
        num_iter += 1

        # Print out the values per iterations
        print('Iteration: {} \t y = {}, x = {}, gradient = {:.4f}'.format(num_iter, obj(xk), xk, l2_norm(grad(xk))))

    # Print the optimal solution of the objective function
    print('\nSolution: \t y = {}, x = {}'.format(objective(xk), xk))

    # Return the x and f(x) in array forms
    return np.array(curve_x), np.array(curve_y)


# Initiate algorithm using the selected parameters
sol, val = newton_method(objective, gradient, hessian)
print(sol, val)

# Plot the learning path of the adagrad gradient descent algorithm in a contour plot
bounds = asarray([[-4.0, 4.0], [-4.0, 4.0]])
xaxis = arange(bounds[0,0], bounds[0,1], 0.1)
yaxis = arange(bounds[1,0], bounds[1,1], 0.1)

# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)

# compute targets
results = objective([x,y])

# create a filled contour plot with 50 levels and jet color scheme
plt.contourf(x, y, results, levels=50, cmap='jet')

# plot the sample as black circles
solution = asarray(sol)
plt.plot(solution[:, 0], solution[:, 1], '.-', color='w')

# show the plot
plt.rcParams["figure.figsize"] = (24,6)
plt.tick_params(axis='y', labelsize=20)
plt.tick_params(axis='x', labelsize=20)
plt.title('Newton Method', fontsize=20)
plt.show()
