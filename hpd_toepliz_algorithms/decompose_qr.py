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
        w = tau * Q[:, k:] @ v  # (m, L) * (L,) -> (m,)
        CNT_GLOB.rmul += 4 * m * L + 2 * m
        CNT_GLOB.radd += 4 * m * L
        # Затем вычитаем tau * outer(w, conj(v))
        # Q[:, k:] = Q[:, k:] - tau * np.outer(w, v.conj())
        Q[:, k:] = Q[:, k:] - np.outer(w, v.conj())
        CNT_GLOB.rmul += 4 * m * L
        CNT_GLOB.radd += 4 * m * L
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


def solve_from_qr(Q, R, b, return_counts=False):
    """
    Solve A x = b with A = Q R.
    y = Q^H b,   then solve R x = y by back substitution.
    """
    n = Q.shape[0]
    cnt = OpCount()
    # y = Q^H @ b
    y = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        s = 0.0j
        for j in range(n):
            s += np.conj(Q[j, i]) * b[j]  # (Q^H)_{i,j} = conj(Q[j,i])
            cnt.rmul += 4
            cnt.radd += 4
        y[i] = s

    # back substitution R x = y  (R upper triangular, diagonal is real)
    x = np.zeros(n, dtype=np.complex128)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= R[i, j] * x[j]
            cnt.rmul += 4
            cnt.radd += 4
        x[i] = s / R[i, i].real
        cnt.rdiv += 2
    return (x, cnt) if return_counts else x


def invert_from_qr_direct(Q, R, return_counts=False):
    """
    Compute A^{-1} by solving R * X = Q^H for X, column by column.
    R is upper triangular with real non-negative diagonal.
    """
    n = Q.shape[0]
    QH = Q.conj().T
    invA = np.empty((n, n), dtype=np.complex128)
    cnt = OpCount()

    for j in range(n):  # для каждого столбца правой части
        x = np.zeros(n, dtype=np.complex128)
        # обратный ход: R x = (столбец j матрицы Q^H)
        for i in range(n - 1, -1, -1):
            s = QH[i, j]
            for k in range(i + 1, n):
                s -= R[i, k] * x[k]
                cnt.rmul += 4  # комплексное умножение
                cnt.radd += 4  # комплексное вычитание + сложения внутри умножения
            # деление на вещественную диагональ
            x[i] = s / R[i, i].real
            cnt.rdiv += 2  # комплексное число / вещественное
        invA[:, j] = x

    return (invA, cnt) if return_counts else invA


def qr_dec_complexity_th(n=8):
    return OpCount(
        rmul=(20 * n**3 + 45 * n**2 + 37 * n) // 3,
        radd=(20 * n**3 + 33 * n**2 + 25 * n) // 3,
        rdiv=5 * n,
        rsqrt=3 * n,
    )


def qr_lsea_complexity_th(n=8):
    return OpCount(rmul=6 * n**2 - 2 * n, radd=6 * n**2 - 2 * n, rdiv=2 * n, rsqrt=0)


def qr_inv_complexity_th(n=8):
    return OpCount(
        rmul=2 * n**2 * (n - 1),
        radd=2 * n**2 * (n - 1),
        rdiv=2 * n**2,  # можно убрать квадрат
        rsqrt=0,
    )


def demo(n=8, display_matrix=False):
    rho = 0.84
    phi = 0.37
    c = np.array(
        [rho**k * np.exp(1j * phi * k) for k in range(n)],
        dtype=np.complex128,
    )
    # A = np.array([[3 + 3j, -1j], [1 + 2j, 2 + 4j]], dtype=np.complex128)
    A = hermitian_toeplitz(c)

    # decomposition
    Q, R = qr_householder(A)
    Q_real, R_real = make_R_diag_real(Q, R)
    Q_np, R_np = np.linalg.qr(A)

    # inversion
    A_inv_qr, inv_qr_counts = invert_from_qr_direct(Q_real, R_real, return_counts=True)
    A_inv_ref = np.linalg.inv(A)
    inv_rel_err = np.linalg.norm(A_inv_qr - A_inv_ref) / np.linalg.norm(A_inv_ref)
    # inv_err = np.max(np.abs(A_inv_qr - A_inv_ref))

    # lsae
    rng = np.random.default_rng(1)
    b = rng.normal(size=n) + 1j * rng.normal(size=n)
    x, lsae_qr_counts = solve_from_qr(Q_real, R_real, b, return_counts=True)
    x_ref = np.linalg.solve(A, b)
    lsae_rel_err = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
    # lsae_err = np.max(np.abs(x - x_ref))

    if display_matrix:
        print("Исходная матрица A:")
        print(A, "\n")

        print("=== Наше QR (стандартное, диагональ R комплексная) ===")
        print("Q:")
        print(Q)
        print("R:")
        print(R)
        print("Q^H Q (проверка унитарности):")
        print(np.round(Q.conj().T @ Q, 14))
        print("A - Q @ R (норма):", np.linalg.norm(A - Q @ R), "\n")

        print("=== После приведения к вещественной диагонали R ===")
        print("Q_real:")
        print(Q_real)
        print("R_real:")
        print(R_real)

        # Сравнение с numpy
        print("\n=== numpy.linalg.qr ===")
        print("Q_np:")
        print(Q_np)
        print("R_np:")
        print(R_np)

    # Проверка совпадения с numpy
    print("\nСравнение с numpy:")
    print("Максимальное отличие Q:", np.max(np.abs(Q_real - Q_np)))
    print("Максимальное отличие R:", np.max(np.abs(R_real - R_np)))
    print("Макс. отклонение |A-QR|: ", np.max(np.abs(A - Q_real @ R_real)))
    print("Ошибка в решении Ax=b: ", lsae_rel_err)
    print("Ошибка A^-1: ", inv_rel_err)

    print("Число операций разложение - измерено: ", CNT_GLOB, " | теория: ", qr_dec_complexity_th(n))
    print("Число операций СЛАУ - измерено: ", lsae_qr_counts, " | теория: ", qr_lsea_complexity_th(n))
    print("Число операций инверсия - измерено: ", inv_qr_counts, " | теория: ", qr_inv_complexity_th(n))


if __name__ == "__main__":
    demo(3)
