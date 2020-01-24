# import random
import numpy as np
import matplotlib.pyplot as plt
import math
from math import sqrt, cos, sin
import clustering

radius = np.random.uniform(low=0, high=1, size=1000)
angle = np.random.uniform(low=0, high=2*math.pi, size=1000)

radius[400:750] = radius[400:750] + 5
radius[750:1000] = radius[750:1000] + 10

points = np.zeros((1000, 2))

for i in range(0, 1000):
    points[i] = np.array((radius[i]*cos(angle[i]), radius[i]*sin(angle[i])))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(points[:, 0], points[:, 1])
plt.show()


def gaussian(x):
    return math.exp(-x*x)


A = np.zeros((1000, 1000))

for i in range(0, 1000):
    for j in range(0, 1000):
        dist = sqrt((points[i, 0] - points[j, 0])**2 +
                    (points[i, 1]-points[j, 1])**2)
        A[i, j] = gaussian(dist)

plt.matshow(A)
plt.colorbar()
plt.show()

X = clustering.clustering(A, 3, 0.1)

for i in range(0, 1000):
    if (X[i, 0] == 1):
        plt.plot(points[i, 0], points[i, 1], 'bo')
    elif (X[i, 1] == 1):
        plt.plot(points[i, 0], points[i, 1], 'ro')
    elif (X[i, 2] == 1):
        plt.plot(points[i, 0], points[i, 1], 'go')

plt.show()
