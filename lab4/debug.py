import numpy as np
from scipy.linalg import schur, svd

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

def svd_via_schur(A: np.ndarray):
    m, n = A.shape
    B, Ut, V = bidiagonalize(A)
    print(B)

    C = np.zeros((m+n, m+n))
    # C[:m, m:] = B
    # C[m:, :m] = B.T
    C[:m, m:] = A
    C[m:, :m] = A.T
    
    S_full, Q = schur(C)
    S = S_full#[:min(m,n), :min(m,n)]

    U = Ut @ Q[:m, :m]

    V = V @ Q[m:, :n]
    
    return U, S, V.T

# Пример использования
if __name__ == "__main__":
    # Пример матрицы из книги (раздел 5.4.8)
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])
    # A = np.array([[1, 1, 0, 0],
    #               [0, 2, 1, 0],
    #               [0, 0, 3, 1],
    #               [0, 0, 0, 4]])
    
    B, Ut, V = bidiagonalize(A)
    
    print("Исходная матрица A:")
    print(A)
    
    print("\nБидиагональная матрица B:")
    print(np.round(B, 6))

    print("\nОртогональная матрица Ut:")
    print(np.round(Ut, 6))

    print("\nОртогональная матрица V:")
    print(np.round(V, 6))
    
    print("\nПроверка ортогональности Ut:")
    print(np.round(Ut.T @ Ut, 6))
    
    print("\nПроверка ортогональности V:")
    print(np.round(V.T @ V, 6))
    
    print("\nПроверка разложения B = Ut A V:")
    print(np.round(Ut @ A @ V, 6))

    A = np.array([[1, 6, 11],
                  [2, 7, 12],
                  [3, 8, 13],
                  [4, 9, 14],
                  [5, 10,15]])
    U, S, V = svd_via_schur(A)
    U_th, s_th, Vt_th = svd(A)
    print("S:")
    print(np.round(S, 4))
    print("s_th:")
    print(np.round(s_th, 4))
    # print("U:")
    # print(U)
    # print("U_th:")
    # print(U_th)