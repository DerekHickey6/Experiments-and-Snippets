from matplotlib import pyplot as plt
import numpy as np

ax = plt.axes(projection="3d")

x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)

X, Y = np.meshgrid(x, y)

Z = np.sin(X) * np.cos(Y)

ax.plot_surface(X, Y, Z, cmap="Spectral")
ax.set_title("3D Plot")
plt.savefig("3D_surface")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


plt.show()