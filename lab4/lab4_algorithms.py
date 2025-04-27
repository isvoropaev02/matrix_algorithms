import numpy as np
from scipy.linalg import norm, qr

def power_iteration(A: np.ndarray, max_iter: int=50, tol=1e-8):
    x_n = np.random.rand(A.shape[1])
    x_n = x_n / norm(x_n)
    for _ in range(max_iter):
        x_n1 = np.dot(A, x_n)
        eigval = norm(x_n1)
        x_n1_norm = x_n1 / eigval
        if norm(x_n1_norm - x_n) < tol:
            break
        x_n = x_n1_norm
    return eigval, x_n # eigenvalue, eigenvector

def schur_decomposition_power_method(A: np.ndarray, max_iter: int=50, tol=1e-8):
    n = A.shape[0]
    U = np.eye(n)
    R = A.copy()
    for k in range(n-1):
        # 1. Используем степенной метод для нахождения k-го собственного вектора
        _, v = power_iteration(R[k:, k:], max_iter=max_iter, tol=tol)
        
        # 2. Дополняем до ортонормированного базиса (ортогонализация с помощью QR-разложения)
        V = np.eye(n-k)
        V[:, 0] = v
        U_k, _ = qr(V)
        
        # 3. Преобразование подобия
        U_ext = np.eye(n)
        U_ext[k:, k:] = U_k
        
        U = U @ U_ext
        R = U_ext.T @ R @ U_ext
    return U, R

def orthogonal_iteration(A, p=None, max_iter: int=50, tol=1e-8):
    # p - количество искомых собственных значений
    n = A.shape[0]
    if p == None:
        p = n
    Q = np.random.randn(n, p)
    Q, _ = qr(Q, mode='economic')
    
    for _ in range(max_iter):
        Z = A @ Q
        Q_new, _ = qr(Z, mode='economic')
        
        err = np.linalg.norm(Q_new - Q, ord='fro')
        if err < tol:
            break
            
        Q = Q_new

    R = Q.T @ A @ Q
    eigvals = np.diag(R)
    
    return eigvals, Q