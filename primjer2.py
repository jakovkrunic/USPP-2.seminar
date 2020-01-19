import random
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import clustering
# 400 tocaka u ravnini, grupirane oko (0.01, 0), (-0.01, 0), (0, 0.01) i (0, -0.01)

points = np.random.random((400, 2)) / 100

for i in range(0, 100):
    points[i][0] = points[i][0] + 0.005
    points[i][1] = points[i][1] - 0.005

for i in range(100, 200):
    points[i][0] = points[i][0] - 0.015
    points[i][1] = points[i][1] - 0.005

for i in range(200, 300):
    points[i][1] = points[i][1] + 0.005
    points[i][0] = points[i][0] - 0.005

for i in range(300, 400):
    points[i][1] = points[i][1] - 0.015
    points[i][0] = points[i][0] - 0.005

points_x = points[:,0]
points_y = points[:,1]

#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)

#ax.scatter(points_x, points_y)
#plt.show()

A = np.zeros((400,400))

for i in range(0, 400):
    for j in range(i+1, 400):
        A[i, j] = 1 / sqrt((points_x[i] - points_x[j]) * (points_x[i] - points_x[j]) +
                           (points_y[i]-points_y[j]) * (points_y[i]-points_y[j]))
        A[j, i] = A[i, j]

A = A / (A.max())

for i in range(0, 400):
    A[i, i] = 1

#plt.matshow(A)
#plt.colorbar()
#plt.show()

X = clustering.clustering(A, 4, 1)

for i in range(0, 400):
    if (X[i, 0] == 1):
        plt.plot(points_x[i], points_y[i], 'bo')
    elif (X[i, 1] == 1):
        plt.plot(points_x[i], points_y[i], 'ro')
    elif (X[i, 2] == 1):
        plt.plot(points_x[i], points_y[i], 'go')
    else:
        plt.plot(points_x[i], points_y[i], 'co')

plt.show()


