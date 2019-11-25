import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.interpolate import griddata

FIG = 1

x1 = x2 = np.linspace(-10, 10, 100)

X1, X2 = np.meshgrid(x1, x2)

Z1 = 6 + (2 * np.power(X1, 2)) + (2 * np.power(X2, 2))
Z2 = 8 + (0 * np.power(X1, 2)) + (0 * np.power(X2, 2))

fig = plt.figure()
ax = fig.gca(projection='3d')

if (FIG == 1):
	surf1 = ax.plot_surface(X=X1, Y=X2, Z=Z1, rstride=3, cstride=3, linewidth=1, cmap=mpl.cm.viridis)   # 3d plot
elif (FIG == 2):
	surf2 = ax.plot_surface(X=X1, Y=X2, Z=Z2, rstride=3, cstride=3, linewidth=1, cmap=mpl.cm.viridis)   # 3d plot

plt.show()
