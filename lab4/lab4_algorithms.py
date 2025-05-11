import numpy as np
from scipy.linalg import norm, qr, hessenberg, schur

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
    return R, U

def orthogonal_iteration(A: np.ndarray, p=None, max_iter: int=50, tol=1e-8):
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


def qr_algorithm_with_shifts(A: np.ndarray, max_iter: int=50):
    n = A.shape[0]
    H, Q = hessenberg(A, calc_q=True)
    R_k = H
    
    for k in range(max_iter):
        # Выбор сдвига (элемент r_nn)
        shift = R_k[-1, -1]

        Q_k, R_k = qr(H - shift * np.eye(n))

        H = R_k @ Q_k + shift * np.eye(n)

        Q = Q @ Q_k
    
    return H, Q


def svd_via_schur(A: np.ndarray, tol=1e-10):
    m, n = A.shape
    
    # Шаг 1: Приведение к бидиагональной форме
    # (Здесь для простоты используем встроенную функцию)
    B, Ut, V = bidiagonalize(A)
    
    # Шаг 2: Построение блочной матрицы
    C = np.zeros((m+n, m+n))
    C[:m, m:] = B
    C[m:, :m] = B.T
    
    # Шаг 3: Разложение Шура для блочной матрицы
    S, Q = schur(C)
    
    # Шаг 4: Извлечение сингулярных значений
    # S = np.abs(np.diag(T)[:min(m,n)])
    # S.sort()[::-1]  # Сортировка по убыванию
    
    # Шаг 5: Извлечение сингулярных векторов
    U = Ut @ Q[:m, :m]
    V = V @ Q[m:, :n]
    
    return U, S, V.T

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
