import sympy as sym


x1 = sym.Symbol('x1')
x2 = sym.Symbol('x2')
x3 = sym.Symbol('x3')
x4 = sym.Symbol('x4')
x5 = sym.Symbol('x4')
w3 = sym.Symbol('w3')
w4 = sym.Symbol('w4')
w5 = sym.Symbol('w5')

t5 = 2*x4 - x3 + w5
t4 = 2*x3 - x2 + w4
t3 = 2*x2 - x1 + w3
dt5 = t5 - t4
dt4 = t4 - t3
ddt5 = dt5 - dt4
pass
