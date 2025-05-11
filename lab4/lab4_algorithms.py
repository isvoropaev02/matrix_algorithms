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
    """
    Вычисление SVD через разложение Шура без явного вычисления AA^T/A^TA
    
    Параметры:
        A : array_like - входная матрица (m x n)
        tol : float - допуск сходимости
        
    Возвращает:
        U, S, V : сингулярные векторы и значения
    """
    m, n = A.shape
    
    # Шаг 1: Приведение к бидиагональной форме
    # (Здесь для простоты используем встроенную функцию)
    B, U, V = bidiagonalize(A)
    
    # Шаг 2: Построение блочной матрицы
    C = np.zeros((m+n, m+n))
    C[:m, m:] = B
    C[m:, :m] = B.T
    
    # Шаг 3: Разложение Шура для блочной матрицы
    T, Q = schur(C)
    
    # Шаг 4: Извлечение сингулярных значений
    S = np.abs(np.diag(T)[:min(m,n)])
    S.sort()[::-1]  # Сортировка по убыванию
    
    # Шаг 5: Извлечение сингулярных векторов
    U = U @ Q[:m, :m]
    V = V @ Q[m:, :n]
    
    return U, S, V.T

def bidiagonalize(A: np.ndarray):
    """Приведение к бидиагональной форме (упрощенная реализация)"""
    m, n = A.shape
    U = np.eye(m)
    V = np.eye(n)
    B = A.copy()
    
    for i in range(min(m,n)):
        # Преобразование Хаусхолдера слева
        x = B[i:, i]
        h = np.eye(m-i) - 2*np.outer(x,x)/np.dot(x,x)
        B[i:, i:] = h @ B[i:, i:]
        U[:, i:] = U[:, i:] @ h.T
        
        if i < n-2:
            # Преобразование Хаусхолдера справа
            x = B[i, i+1:]
            h = np.eye(n-i-1) - 2*np.outer(x,x)/np.dot(x,x)
            B[i:, i+1:] = B[i:, i+1:] @ h.T
            V[i+1:, :] = h @ V[i+1:, :]
    
    return B, U, V
