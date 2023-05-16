import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def func(x):
    f = (x[0] + 3) ** 2 + np.sin(x[0]) + (x[1] + 1) ** 2
    return f


def func_grad(x):
    f = (x[0] + 3) ** 2 + np.sin(x[0]) + (x[1] + 1) ** 2
    grad = np.array([2 * (x[0] + 3) + np.cos(x[0]), 2 * (x[1] + 1)])
    return f, grad


def main():
    x0 = np.zeros(2)

    # gradient descent without gradient function provided -> more computationally expensive
    xopt, fopt, info = fmin_l_bfgs_b(func, x0, approx_grad=True)
    print(xopt, fopt, info)

    # gradient descent with gradient function provided -> more efficient
    xopt, fopt, info = fmin_l_bfgs_b(func_grad, x0)
    print(xopt, fopt, info)


if __name__ == "__main__":
    main()
