import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.linalg import eigh
import random

def clustering(W, K, tolerance):
    # Prvi korak: nadji dijagonalu
    N = W.shape[0]
    ones_N = np.ones((N, 1))
    D = np.diag(np.transpose(np.matmul(W, ones_N)).tolist()[0])

    # Drugi korak: nadji D^(-1/2), svojstvene vrijednosti i vektore,
    #              odredi Z
    D_power = fractional_matrix_power(D, -0.5)

    S, V = eigh(np.matmul(np.matmul(D_power, W),
                          D_power),
                eigvals=(N-K, N-1))

    Z = np.matmul(D_power, V)

    # Treci korak: normaliziranje
    umnozak_Z = np.matmul(Z, np.transpose(Z))
    dijagonala_umnoska_Z = np.diag(np.diag(umnozak_Z))
    dijagonala_power = fractional_matrix_power(dijagonala_umnoska_Z, -0.5)
    X_crta = np.matmul(dijagonala_power, Z)

    # Cetvrti korak
    R = np.zeros((K, K))
    i = random.randint(0, N-1)
    R[:,0] = np.transpose(X_crta[i,:])
    c = np.zeros((N, 1))
    for k in range(1, K):
        c = c + np.abs(np.matmul(X_crta, R[:, k-1]))
        i = np.argmin(c)
        R[:, k] = np.transpose(X_crta[i,:])

    # peti korak
    par_conv = 0
    X = np.zeros((N, K))
    for it in range(1, 100000):
        # sesti korak
        X2 = np.matmul(X_crta, R)
        for i in range(0, N):
            l = np.argmax(X2[i,:])
            X[i, :] = np.zeros((1, K))
            X[i, l] = 1

        u, s, vh = np.linalg.svd(np.matmul(np.transpose(X), X_crta), full_matrices=True)
        v = np.transpose(vh)
        phi = np.sum(s)
        if (abs(phi - par_conv) < tolerance):
            break
        par_conv = phi
        R = np.matmul(v, np.transpose(u))

    return X
 
#W = np.random.rand(5,5)
#clustering(W, 2, 1)
