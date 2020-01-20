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

rows = M.shape[0]
columns = M.shape[1]

M = M.reshape((rows*columns, 3))

#print(M)

n = rows * columns
print(n)
A = np.zeros((n,n), dtype=np.float32)
#A = A.reshape((n,n))

for i in range(0, n):
    for j in range(0, n):
       # print(j)
        A[i,j] = np.linalg.norm(M[i]-M[j])
        A[i,j] = A[i,j]**2
        A[i,j] = math.exp(-A[i,j])
    print('+++' + str(i) + '+++')

X = clustering.clustering(A, 3, 1)

M = M.reshape((rows, columns, 3))

for i in range(0, n):
    a = i // columns
    b = i % columns

    stavi = np.argmax(X[i, :])
    M[a,b,0] = M[a,b,1] = M[a,b,2] = 0
    M[a,b,stavi] = 255

cv2.imshow('AWAW morzek', M)
