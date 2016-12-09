import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-3, 3, 250)
x, y = np.meshgrid(x, x)

f = lambda x, y: np.array([x - y**2 + 1.28 + 1.4*x*y , 0.2*y - x + x**3])

xp = np.zeros_like(x)
yp = np.zeros_like(y)

xp[:], yp[:] = f(x[:], y[:])

magnitude = np.sqrt(xp**2 + yp**2)

fig = plt.figure()
ax = fig.gca()

ax.streamplot(x, y, xp, yp, color=magnitude, cmap=plt.cm.inferno_r)
ax.contour(x, y, xp, levels=[0.], colors='k')
ax.contour(x, y, yp, levels=[0.], colors='k')
ax.axis('equal')
plt.show()
