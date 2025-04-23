import numpy as np
import time
from numba import cuda, float64

# Load data with 64-bit precision
A = np.loadtxt('/home/taosimin/iEDA24/iEDA/bin/matrix.txt', dtype=np.float64)
current = np.loadtxt('/home/taosimin/iEDA24/iEDA/bin/current.txt', dtype=np.float64)
init = np.ones_like(current, dtype=np.float64) * 0.81 * 0.9

# Define a kernel function for matrix-vector multiplication
@cuda.jit
def matvec_kernel(A, x, b, N):
    i = cuda.grid(1)
    if i < N:
        sum = 0.0
        for j in range(N):
            sum += A[i, j] * x[j]
        b[i] = sum

# Define a kernel function for vector subtraction
@cuda.jit
def subtract_kernel(a, b, c, N):
    i = cuda.grid(1)
    if i < N:
        c[i] = a[i] - b[i]

# Define a kernel function for dot product
@cuda.jit
def dot_product_kernel(a, b, result, N):
    sdata = cuda.shared.array(256, dtype=float64)  # Fixed shared memory size
    tid = cuda.threadIdx.x
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    sum = 0.0
    if i < N:
        sum = a[i] * b[i]
    sdata[tid] = sum
    cuda.syncthreads()

    # Perform reduction in shared memory
    s = 256 // 2
    while s > 0:
        if tid < s:
            sdata[tid] += sdata[tid + s]
        cuda.syncthreads()
        s //= 2

    if tid == 0:
        cuda.atomic.add(result, 0, sdata[0])

# Define the CG solver function
def cg_solver(A, b, init=None, tol=1e-6, maxiter=100):
    N = b.shape[0]
    if init is None:
        X = np.zeros(N, dtype=np.float64)
    else:
        X = np.array(init, dtype=np.float64)
        if not (X.shape == (N, 1) or X.shape == (N,)):
            raise ValueError('shapes of A {} and x0 {} are incompatible'
                           .format(A.shape, X.shape))
    X = X.ravel()
    
    # Allocate memory on the GPU
    d_A = cuda.to_device(A)
    d_b = cuda.to_device(b)
    d_X = cuda.to_device(X)
    d_residual = cuda.device_array_like(d_b)
    d_p = cuda.device_array_like(d_b)
    d_Ap = cuda.device_array_like(d_b)
    d_result = cuda.device_array(1, dtype=np.float64)
    
    # Compute initial residual: residual = b - A @ X
    threads_per_block = 256
    blocks_per_grid = (N + (threads_per_block - 1)) // threads_per_block
    matvec_kernel[blocks_per_grid, threads_per_block](d_A, d_X, d_residual, N)
    subtract_kernel[blocks_per_grid, threads_per_block](d_b, d_residual, d_residual, N)
    
    d_p.copy_to_device(d_residual)
    d_result[0] = 0.0
    dot_product_kernel[blocks_per_grid, threads_per_block](d_residual, d_residual, d_result, N)
    cuda.synchronize()
    r_dot_r = d_result.copy_to_host()[0]
    
    for i in range(maxiter):
        d_result[0] = 0.0
        matvec_kernel[blocks_per_grid, threads_per_block](d_A, d_p, d_Ap, N)
        
        d_result[0] = 0.0
        dot_product_kernel[blocks_per_grid, threads_per_block](d_p, d_Ap, d_result, N)
        cuda.synchronize()
        alpha = r_dot_r / d_result.copy_to_host()[0]
        
        d_X += alpha * d_p
        d_residual -= alpha * d_Ap
        
        d_result[0] = 0.0
        dot_product_kernel[blocks_per_grid, threads_per_block](d_residual, d_residual, d_result, N)
        cuda.synchronize()
        r_dot_r_new = d_result.copy_to_host()[0]
        beta = r_dot_r_new / r_dot_r
        d_p = d_residual + beta * d_p
        r_dot_r = r_dot_r_new
        
        if r_dot_r < tol:
            break
    
    X = d_X
    return X, i, r_dot_r

start_time = time.time()
X, num_iter, r_dot_r = cg_solver(A, current, init=init, tol=1e-6, maxiter=1000)
print("X: ", X)
print("num_iter: ", num_iter)
print("r_dot_r: ", np.sqrt(r_dot_r))

residual = np.linalg.norm(A @ X - current)
print(f"residual: {residual}")

end_time = time.time()

execution_time = end_time - start_time
print("cg solver execution time: {:.6f} seconds".format(execution_time))