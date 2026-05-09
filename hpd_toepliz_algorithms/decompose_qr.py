import numpy as np
from utilities import OpCount, hermitian_toeplitz

CNT_GLOB = OpCount()


def householder_reflection(x):
    """Возвращает вектор v и число tau, такие что H = I - tau * v @ v^H
    обнуляет все элементы x, кроме первого, делая его равным -phase(x0)*norm(x)."""
    L = len(x)
    # Норма
    norm_x = np.linalg.norm(x)
    CNT_GLOB.rmul += 2 * L
    CNT_GLOB.radd += 2 * L
    CNT_GLOB.rsqrt += 1
    if norm_x == 0:
        # тривиальный случай
        v = np.zeros(L, dtype=np.complex128)
        v[0] = 1.0
        tau = 0.0
        return v, tau
    # Фаза первого элемента
    x0 = x[0]
    if abs(x0) > 0:
        phase = x0 / abs(x0)
    else:
        phase = 1.0 + 0.0j
    CNT_GLOB.rdiv += 2
    CNT_GLOB.rmul += 2
    CNT_GLOB.radd += 1
    CNT_GLOB.rsqrt += 1
    # alpha = -phase * norm_x
    alpha = -phase * norm_x
    CNT_GLOB.rmul += 2
    # v = x - alpha * e1
    v = x.copy()
    v[0] = x0 - alpha
    CNT_GLOB.radd += 2
    # tau = 2 / (v^H v)
    v_norm_sq = np.vdot(v, v).real
    CNT_GLOB.rmul += 2 * L
    CNT_GLOB.radd += 2 * L
    tau = 2.0 / v_norm_sq if v_norm_sq != 0 else 0.0
    CNT_GLOB.rdiv += 1
    return v, tau


def qr_householder(A):
    """QR-разложение Хаусхолдера. Возвращает Q (унитарная), R (верхнетреугольная)."""
    m, n = A.shape
    Q = np.eye(m, dtype=np.complex128)
    R = A.astype(np.complex128).copy()
    min_mn = min(m, n)

    for k in range(min_mn):
        L = m - k
        x = R[k:, k].copy()
        v, tau = householder_reflection(x)

        # Применяем отражение к подматрице R[k:, k:]
        for j in range(k, n):
            col = R[k:, j]
            # col = col - tau * (v^H col) * v
            dot = np.vdot(v, col)  # комплексное число
            CNT_GLOB.rmul += 4 * L
            CNT_GLOB.radd += 4 * L
            R[k:, j] = col - (tau * dot) * v
            CNT_GLOB.rmul += 4 * L + 2
            CNT_GLOB.radd += 4 * L

        # Накопление Q: умножаем справа на H_k
        # Q_new = Q_old - tau * (Q_old[:, k:] @ v) @ v^H
        # Сначала вычисляем w = Q[:, k:] @ v  (вектор длины m)
        # Q[:, k:] имеет размер m x L, v длины L
        w = Q[:, k:] @ v  # (m, L) * (L,) -> (m,)
        CNT_GLOB.rmul += 4 * m * L
        CNT_GLOB.radd += 4 * m * L
        # Затем вычитаем tau * outer(w, conj(v))
        Q[:, k:] = Q[:, k:] - tau * np.outer(w, v.conj())
        CNT_GLOB.rmul += 6 * m * L
        CNT_GLOB.radd += 6 * m * L

    return Q, R


def make_R_diag_real(Q, R):
    """Приводит R к вещественной диагонали, как в numpy.linalg.qr.
    Возвращает новые Q, R."""
    diag = np.diag(R)
    L = len(diag)
    # Вычисляем фазы d_i так, чтобы d_i * R_ii было вещественным (в numpy оно отрицательное)
    # d_i = - conj(R_ii) / |R_ii|
    d = np.zeros(L, dtype=np.complex128)
    for i, rii in enumerate(diag):
        if abs(rii) > 1e-15:
            d[i] = -np.conj(rii) / abs(rii)
        else:
            d[i] = 1.0 + 0.0j
        CNT_GLOB.rdiv += 2
        CNT_GLOB.rmul += 2
        CNT_GLOB.radd += 1
        CNT_GLOB.rsqrt += 1
    D = np.diag(d)
    R_new = D @ R
    Q_new = Q @ D.conj().T  # D унитарна, поэтому Q_new унитарна
    CNT_GLOB.rmul += 4 * ((L**2 + L) // 2)
    CNT_GLOB.radd += 2 * ((L**2 + L) // 2)
    return Q_new, R_new


# ================================================
# Тест для заданной матрицы 2x2
# A = np.array([[3 + 3j, -1j], [1 + 2j, 2 + 4j]], dtype=np.complex128)
# A = np.array([[2 + 6j, 2 - 1j], [1 + 2j, 1 - 0.5j]], dtype=np.complex128)
# A = np.array([[2 + 6j, 2 - 1j, 2], [1 + 2j, 1 - 0.5j, -1 - 3j], [0, -0.2j, 1 - 0.2j]], dtype=np.complex128)
# A = np.random.randint(low=-5, high=5, size=(7, 7))


def demo(n=8):
    rho = 0.80
    phi = 0.37
    c = np.array(
        [rho**k * np.exp(1j * phi * k) for k in range(n)],
        dtype=np.complex128,
    )
    A = hermitian_toeplitz(c)

    print("Исходная матрица A:")
    print(A, "\n")

    # Наше разложение
    Q, R = qr_householder(A)
    print("=== Наше QR (стандартное, диагональ R комплексная) ===")
    print("Q:")
    print(Q)
    print("R:")
    print(R)
    print("Q^H Q (проверка унитарности):")
    print(np.round(Q.conj().T @ Q, 14))
    print("A - Q @ R (норма):", np.linalg.norm(A - Q @ R), "\n")

    # Приведение к вещественной диагонали
    Q_real, R_real = make_R_diag_real(Q, R)
    print("=== После приведения к вещественной диагонали R ===")
    print("Q_real:")
    print(Q_real)
    print("R_real:")
    print(R_real)
    print(np.max(np.abs(A - Q_real @ R_real)))

    # Сравнение с numpy
    Q_np, R_np = np.linalg.qr(A)
    print("\n=== numpy.linalg.qr ===")
    print("Q_np:")
    print(Q_np)
    print("R_np:")
    print(R_np)

    # Проверка совпадения с numpy
    print("\nСравнение с numpy:")
    print("Максимальное отличие Q:", np.max(np.abs(Q_real - Q_np)))
    print("Максимальное отличие R:", np.max(np.abs(R_real - R_np)))

    print("Число операций: ", CNT_GLOB)


if __name__ == "__main__":
    demo(3)
