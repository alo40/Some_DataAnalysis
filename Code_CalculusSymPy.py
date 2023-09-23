from sympy import *
# init_printing()

x = Symbol('x')
density = 105 * x**2 * (1 - x)**4
total_area = Integral(density, (x, 0, 1))
print(total_area.doit())

prob_02_04 = Integral(density, (x, 0.2, 0.4)).doit()
print(prob_02_04)



import numpy as np
import matplotlib.pyplot as plt

# x = np.linspace(0, 1, 100)
# f = 105 * x**2 * (1 - x)**4
#
# plt.plot(x, f)
# plt.show()