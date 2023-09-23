from sympy import symbols, solve, exp, log, lambdify, sqrt
from scipy.optimize import fsolve

a = symbols('a', positive=True)

# using normal distribution (SNE)
x = exp(-1**2)
epsilon = exp(-sqrt(2)**2)
# y = exp(-a**2)
# delta = exp(-(2 * a)**2)

# using t-distribution (t-SNE) - only for y's
y = (1 + a**2)**(-1)
delta = (1 + (2 * a)**2)**(-1)

p12 = x / (2 * x + epsilon)
p23 = epsilon / (2 * x + epsilon)
q12 = y / (2 * y + delta)
q23 = delta / (2 * y + delta)
KL = 2 * p12 * log(p12 / q12) + p23 * log(p23 / q23)
# sol = solve(KL)  # symbolical solution not possible


func_np = lambdify(a, KL, modules=['numpy'])
solution = fsolve(func_np, 1.)
print(solution)
