import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define the range for x and y values
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)

# Create a grid of (x, y) values
X, Y = np.meshgrid(x, y)

# Calculate the function values f = x * y
F = -X**4 - Y**4

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, F, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3)

# Add labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('F')

# # Add a colorbar
# fig.colorbar(surf, label='f = x * y')

# Show the 3D plot
# plt.title('3D Plot of f = x * y')
plt.show()