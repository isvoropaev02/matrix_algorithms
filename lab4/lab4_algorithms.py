import numpy as np
from scipy.linalg import norm, qr, hessenberg, schur

def power_iteration_classic(A: np.ndarray, max_iter: int=50):
    x_n = np.random.rand(A.shape[1])
    x_n = x_n / norm(x_n)
    for _ in range(max_iter):
        x_n1 = np.dot(A, x_n)
        eigval = norm(x_n1)
        x_n1_norm = x_n1 / eigval
        x_n = x_n1_norm
    return eigval, x_n # eigenvalue, eigenvector

def power_iteration_with_atol(A: np.ndarray, max_iter: int=50, atol=1e-8):
    x_n, w_n = np.random.rand(A.shape[1]), np.random.rand(A.shape[0])
    x_n = x_n / norm(x_n)
    w_n = w_n / norm(w_n)
    for _ in range(max_iter):
        x_n1, w_n1 = np.dot(A, x_n), np.dot(A.T, w_n)
        eigval = norm(x_n1)
        x_n, w_n = x_n1 / eigval, w_n1 / norm(w_n1)
        r_n, s_n = norm(np.dot(A, x_n)-eigval*x_n), np.inner(w_n, x_n)
        if r_n / s_n < atol:
            break
    return eigval, x_n # eigenvalue, eigenvector

def orthogonal_iteration(A: np.ndarray, p=None, max_iter: int=50):
    # p - количество искомых собственных значений
    n = A.shape[0]
    if p == None:
        p = n
    Q, _ = qr(np.random.randn(n, p), mode='economic')
    for _ in range(max_iter):
        Z = A @ Q
        Q_new, _ = qr(Z, mode='economic')
        Q = Q_new
    R = Q.T @ A @ Q
    eigvals = np.diag(R)
    return eigvals, Q

def schur_decomposition_power_method(A: np.ndarray, max_iter: int=50):
    assert A.shape[0] == A.shape[1], "A has to be a square matrix"
    n = A.shape[0]
    Q = np.eye(n)
    for _ in range(max_iter):
        Z = np.dot(A, Q)
        Q, R = qr(Z)
    return R, Q

def qr_algorithm_no_shifts(A: np.ndarray, max_iter: int=50):
    assert A.shape[0] == A.shape[1], "A has to be a square matrix"
    n = A.shape[0]
    H, Q = hessenberg(A, calc_q=True)
    R_k = H
    for _ in range(max_iter):
        Q_k, R_k = qr(H)
        H = R_k @ Q_k
        Q = Q @ Q_k
    return H, Q

def qr_algorithm_with_shifts(A: np.ndarray, max_iter: int=50):
    assert A.shape[0] == A.shape[1], "A has to be a square matrix"
    n = A.shape[0]
    H, Q = hessenberg(A, calc_q=True)
    R_k = H
    for _ in range(max_iter):
        # Выбор сдвига (элемент r_nn)
        shift = R_k[-1, -1]
        Q_k, R_k = qr(H - shift * np.eye(n))
        H = R_k @ Q_k + shift * np.eye(n)
        Q = Q @ Q_k
    return H, Q

def svd_via_schur2(A: np.ndarray):
    m, n = A.shape
    # вот сюда бы еще бидиагонализацию A
    C = np.zeros((m+n, m+n))
    C[:m, m:] = A
    C[m:, :m] = A.T
    
    S_full, Q = schur(C)
    col_idx = np.argsort(np.diag(S_full))[::-1]
    s_vals = np.diag(S_full)[col_idx[:min(n, m)]]
    S = np.zeros(shape=(m, n))
    S[:min(m, n), :min(m, n)] = np.diag(s_vals)
    singular_vecs = Q[:, col_idx[:-min(n, m)]]
    U = singular_vecs[:m, :m]
    U /= np.linalg.norm(singular_vecs[:m, :m], axis=0)
    V = singular_vecs[m:, :n]
    V /= np.linalg.norm(singular_vecs[m:, :n], axis=0)
    
    return U, S, V.T, s_vals


#### Эта часть пока не пошла в ход (для улучшения SVD) --------------------------------------------------------------
def householder_vector(x: np.ndarray):
    sign = -1 if x[0] >= 0 else 1
    v = x.copy()
    alpha = np.linalg.norm(x)
    if alpha == 0:
        beta = 0
    else:
        v[0] = v[0] - sign * alpha
        v = v / np.linalg.norm(v)
        beta = 2
    return v, beta

def bidiagonalize(A):
    m, n = A.shape # m >= n should be
    B = A.copy()
    Ut = np.eye(m)
    V = np.eye(n)
    
    for k in range(min(m, n)):
        # Левый преобразователь Хаусхолдера (столбец k)
        if k < m:
            x = B[k:, k]
            if np.linalg.norm(x[1:]) > 0:
                v, beta = householder_vector(x)
                # Применяем преобразование к B
                B[k:, k:] = B[k:, k:] - beta * np.outer(v, v @ B[k:, k:])
                # Аккумулируем преобразование в Ut
                Ut[k:, :] = Ut[k:, :] - beta * np.outer(v, v @ Ut[k:, :])
        
        # Правый преобразователь Хаусхолдера (строка k)
        if k < n - 2:
            x = B[k, k+1:]
            if np.linalg.norm(x[1:]) > 0:
                v, beta = householder_vector(x)
                # Применяем преобразование к B
                B[k:, k+1:] = B[k:, k+1:] - beta * np.outer(B[k:, k+1:] @ v, v)
                # Аккумулируем преобразование в V
                V[k+1:, :] = V[k+1:, :] - beta * np.outer(v, v @ V[k+1:, :])
    
    return B, Ut, V
