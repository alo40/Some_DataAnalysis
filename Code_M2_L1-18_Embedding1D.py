from sympy import symbols, solve, exp, log, lambdify
from scipy.optimize import fsolve

# from sympy import *

a = symbols('a', positive=True)
x = exp(-1)
epsilon = exp(-2)
y = exp(-a ** 2)
delta = exp(-4 * a ** 2)
p12 = x / (2 * x + epsilon)
p23 = epsilon / (2 * x + epsilon)
q12 = y / (2 * y + delta)
q23 = delta / (2 * y + delta)
KL = 2 * p12 * log(p12 / q12) + p23 * log(p23 / q23)
# sol = solve(KL)  # symbolical solution not possible


func_np = lambdify(a, KL, modules=['numpy'])
solution = fsolve(func_np, 0.5)
print(solution)
