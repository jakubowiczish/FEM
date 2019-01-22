import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


# The task is to solve the following differential equation using finite element method
# (a(x) * u'(x))' + b(x) * u'(x) + c(x)* u(x) = f(x)
# with following conditions:
# x ∈ [0,1]
# -a(0) * u'(0) + beta * u(0) = gamma
# u(1) = u1

def a(x): return 1


def b(x): return 2


def c(x): return -1


def f(x): return -x * x + 4 * x + 3


beta = 1
gamma = -1
u1 = 0
n = 50


# Basis function for FEM
def basis_function(k: int):
    return lambda x: max(0.0, (1.0 - abs(x * n - k)))


# The derivative of the basis function
def basis_function_derivative(k):
    def function_derivative(x):
        if basis_function(k)(x) == 0:
            return 0
        elif x * n < k:
            return n
        else:
            return -n

    return function_derivative


# The left side of the equation
def b_u_v(u, v, d_u, d_v, limit_a: float, limit_b: float):
    def first_component(x): return a(x) * d_u(x) * d_v(x)

    def second_component(x): return b(x) * d_u(x) * v(x)

    def third_component(x): return c(x) * u(x) * v(x)

    return (
            - beta * u(0) * v(0)
            - quad(lambda x: first_component(x), limit_a, limit_b)[0]
            + quad(lambda x: second_component(x), limit_a, limit_b)[0]
            + quad(lambda x: third_component(x), limit_a, limit_b)[0]
    )


# The right side of the equation
def l_v(v, limit_a, limit_b):
    def first_component(x): return f(x) * v(x)

    return (
            - gamma * v(0)
            + quad(lambda x: first_component(x), limit_a, limit_b)[0]
    )


# Method that fills the matrix with basis functions
def fill_b_u_v_matrix(i: int, j: int):
    limit_a = min(max(0.0, (i - 1) / n), max(0.0, (j - 1) / n))
    limit_b = max(min(1.0, (j + 1) / n), min(1.0, (i + 1) / n))

    return b_u_v(
        basis_function(i),
        basis_function(j),
        basis_function_derivative(i),
        basis_function_derivative(j),
        limit_a,
        limit_b
    )


# Method returns solution for the FEM
def get_solution():
    left_matrix = np.empty((n, n))

    for i in range(0, n):
        for j in range(0, n):
            left_matrix[i][j] = fill_b_u_v_matrix(j, i)

    def shift_solution(x):
        return basis_function(n)(x) * u1

    right_matrix = np.empty(n)

    for i in range(0, n):
        right_matrix[i] = (
                l_v(
                    basis_function(i),
                    max(0.0, (i - 1) / n),
                    min(1.0, (i + 1) / n)
                )
                -
                b_u_v(
                    shift_solution,
                    basis_function(i),
                    lambda x: basis_function_derivative(n)(x) * u1,
                    basis_function_derivative(i),
                    max(0.0, (i - 1) / n),
                    min(1.0, (i + 1) / n)
                )
        )

    star_solution = np.linalg.solve(left_matrix, right_matrix)

    def calculate_star_solution(x):
        u_star = 0
        for i in range(0, n):
            u_star += star_solution[i] * basis_function(i)(x)

        return u_star

    return lambda x: shift_solution(x) + calculate_star_solution(x)


def main():
    solution = get_solution()
    x_s = np.linspace(0, 1, 50)
    y_s = [solution(x) for x in x_s]

    plt.plot(x_s, y_s)
    plt.show()


if __name__ == '__main__':
    main()
