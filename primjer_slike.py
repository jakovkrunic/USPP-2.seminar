import numpy as np
import matplotlib.pyplot as plt
import math
from math import sqrt, cos, sin
import clustering
import cv2

#M = plt.imread('E:/GitHub/USPP-2.seminar/primjer_slike/walrus1.png', 'png')
#print(M.shape)

M = cv2.imread('E:/GitHub/USPP-2.seminar/primjer_slike/walrus1.png')
#print(M)

plt.imshow(M)
plt.axis('off')
plt.show()

rows = M.shape[0]
columns = M.shape[1]

M = M.reshape((rows*columns, 3))

n = rows * columns
print(n)
A = np.zeros((n,n), dtype=np.float64)
#A = A.reshape((n,n))

for i in range(0, n):
    for j in range(0, n):
       # print(j)
        #print(M[i])
        A[i,j] = np.linalg.norm(M[i]-M[j]) / 50
        #A[i,j] = A[i,j]**2
        A[i,j] = math.exp(-A[i,j])
        #if (A[i,j] != 1.0):
            #print(M[i])
            #print(M[j])
            #print(np.linalg.norm(M[i]-M[j]))
            #print(np.linalg.norm(M[i]-M[j]))
            #print(A[i,j])
    print('+++' + str(i) + '+++')

X = clustering.clustering(A, 3, 0.00001)

M = M.reshape((rows, columns, 3))

for i in range(0, n):
    a = i // columns
    b = i % columns

    stavi = np.argmax(X[i, :])
    M[a,b,0] = M[a,b,1] = M[a,b,2] = 0
    M[a,b,stavi] = 255

plt.imshow(M)
plt.axis('off')
plt.show()
