from scipy.linalg import lu
import numpy as np


A = np.loadtxt('/home/taosimin/iEDA24/iEDA/bin/matrix.txt', dtype=np.float64)
current = np.loadtxt('/home/taosimin/iEDA24/iEDA/bin/current.txt', dtype=np.float64)
init = np.ones_like(current, dtype=np.float64) * 0.81 * 0.9
def cg_solver(A, b, init=None, tol=1e-6, maxiter=100):
    N = len(b)
    if init is None:
        X = np.zeros(N, dtype=float)
    else:
        X = np.array(init, dtype=float)
        if not (X.shape == (N,1) or X.shape == (N,)):
            raise ValueError('shapes of A {} and x0 {} are incompatible'
               .format(A.shape, X.shape))
    X = X.ravel()
    
    residual = A.dot(X)
    print(residual)

    residual = b - residual
    
    print(residual)
    
    p = residual.copy()
    r_dot_r = np.dot(residual, residual)

    for i in range(maxiter):
        Ap = A.dot(p)
        alpha = r_dot_r / np.dot(p, Ap)
        X += alpha * p
        residual -= alpha * Ap

        r_dot_r_new = np.dot(residual, residual)
        beta = r_dot_r_new / r_dot_r
        p = residual + beta * p
        r_dot_r = r_dot_r_new

        if r_dot_r < tol:
            break
    return X, i

X, num_iter = cg_solver(A, current, init=init, tol=1e-6, maxiter=1000)
print("X: ", X)
print("num_iter: ", num_iter)
