import numpy as np
from utilities import OpCount, hermitian_toeplitz


def toeplitz_cholesky_schur(c, return_counts=False):
    """
    Lower Cholesky factor L for Hermitian positive-definite Toeplitz T:
        T = L @ L.conj().T

    This is a Schur/Bareiss-style O(n^2) Toeplitz Cholesky recursion.
    The implementation is intentionally explicit and loop-based.
    """
    c = np.asarray(c, dtype=np.complex128)
    n = len(c)

    if n == 0:
        L = np.empty((0, 0), dtype=np.complex128)
        return (L, OpCount()) if return_counts else L

    if abs(c[0].imag) > 1e-13 or c[0].real <= 0:
        raise ValueError("c[0] must be positive real for HPD Hermitian Toeplitz.")

    cnt = OpCount()

    # General c0 != 1 support.
    s0 = np.sqrt(c[0].real)
    cnt.rsqrt += 1

    # g0, g1 are Schur generator rows scaled so that first Cholesky column is correct.
    g0 = c.astype(np.complex128).copy() / s0
    g1 = g0.copy()
    cnt.rdiv += 2 * n  # two complex vectors divided by real scalar

    L = np.zeros((n, n), dtype=np.complex128)

    # First column.
    for j in range(n):
        L[j, 0] = g0[j]

    # Shift g0 right by one element.
    for j in range(n - 1, 0, -1):
        g0[j] = g0[j - 1]
    g0[0] = 0.0

    for i in range(1, n):
        # In exact arithmetic g0[i] is positive real.
        denom = g0[i].real
        rho = -g1[i] / denom
        cnt.rdiv += 2

        gamma = np.sqrt(1.0 - (rho.real * rho.real + rho.imag * rho.imag))
        cnt.rsqrt += 1
        cnt.rmul += 2
        cnt.radd += 2

        for j in range(i, n):
            alpha = g0[j]
            beta = g1[j]

            # Hyperbolic Schur rotation:
            #   new_g0 = (alpha + conj(rho) * beta) / gamma
            #   new_g1 = (rho * alpha + beta) / gamma
            new_g0 = (alpha + np.conj(rho) * beta) / gamma
            new_g1 = (rho * alpha + beta) / gamma

            cnt.rmul += 8
            cnt.radd += 8
            cnt.rdiv += 4

            g0[j] = new_g0
            g1[j] = new_g1

        for j in range(i, n):
            L[j, i] = g0[j]

        # Shift active part of g0 right by one element.
        for j in range(n - 1, i, -1):
            g0[j] = g0[j - 1]
        g0[i] = 0.0

    return (L, cnt) if return_counts else L


def solve_from_chol_dec(L, b, return_counts=False):
    """
    Solve T x = b where T = L @ L^H, with L lower-triangular.
    Uses forward substitution L y = b and backward substitution L^H x = y.
    L is assumed to have real positive diagonal.
    """
    n = L.shape[0]
    if n == 0:
        x = np.empty(0, dtype=np.complex128)
        return (x, OpCount()) if return_counts else x

    cnt = OpCount()
    y = np.empty(n, dtype=np.complex128)

    # Forward substitution L y = b
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[i, j] * y[j]
            cnt.rmul += 4
            cnt.radd += 4
        # division by real diagonal L[i,i]
        y[i] = s / L[i, i].real
        cnt.rdiv += 2

    # Backward substitution L^H x = y
    x = np.empty(n, dtype=np.complex128)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            # L^H[i,j] = conj(L[j,i])
            s -= np.conj(L[j, i]) * x[j]
            cnt.rmul += 4
            cnt.radd += 4
        x[i] = s / L[i, i].real  # L^H[i,i] = L[i,i] (real)
        cnt.rdiv += 2

    return x, cnt if return_counts else x


def invert_from_chol_dec(L, return_counts=False):
    """
    Compute T^{-1} from Cholesky factor L (T = L @ L^H).
    Returns the dense inverse matrix.
    Complexity: O(n^3) operations.
    """
    n = L.shape[0]
    if n == 0:
        invT = np.empty((0, 0), dtype=np.complex128)
        return (invT, OpCount()) if return_counts else invT

    cnt = OpCount()
    # M will be L^{-1} (lower triangular)
    M = np.zeros((n, n), dtype=np.complex128)

    # Compute M column by column
    for k in range(n):
        # Elements above diagonal are zero
        # Diagonal and below: solve L * M[:, k] = e_k
        for i in range(k, n):
            s = 0.0
            for j in range(k, i):
                s += L[i, j] * M[j, k]
                cnt.rmul += 4
                cnt.radd += 4
            # s = sum_{j=k}^{i-1} L[i,j] M[j,k]
            # right-hand side is 1 if i==k else 0
            rhs = 1.0 if i == k else 0.0
            M[i, k] = (rhs - s) / L[i, i].real
            cnt.radd += 2
            cnt.rdiv += 2

    # Now T^{-1} = M^H @ M
    invT = np.empty((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            # Compute (i,j) entry: sum_{t=max(i,j)}^{n-1} conj(M[t,i]) * M[t,j]
            # since M is lower triangular, M[t,i]=0 for t<i, M[t,j]=0 for t<j
            s = 0.0j
            start = max(i, j)
            for t in range(start, n):
                s += np.conj(M[t, i]) * M[t, j]
                cnt.rmul += 4
                cnt.radd += 4
            invT[i, j] = s

    return (invT, cnt) if return_counts else invT


def chol_dec_th(n=8):
    return OpCount(rmul=(2 * (n - 1) * (2 * n + 1)), radd=(2 * (n - 1) * (2 * n + 1)), rdiv=(2 * (n**2 + n - 1)), rsqrt=n)


def solve_chol_th(n=8):
    return OpCount(rmul=(4 * n * (n - 1)), radd=(4 * n * (n - 1)), rdiv=(4 * n), rsqrt=0)


def inv_chol_th(n=8):
    return OpCount(rmul=(2 * n**2 * (n + 1)), radd=(n * (n + 1) * (2 * n + 1)), rdiv=(n * (n + 1)), rsqrt=0)


def demo(n=8):
    # Positive-definite Hermitian Toeplitz example:
    # c[k] = rho^k * exp(i phi k), |rho| < 1.
    rho = 0.89
    phi = 0.37
    c = np.array(
        [rho**k * np.exp(1j * phi * k) for k in range(n)],
        dtype=np.complex128,
    )

    T = hermitian_toeplitz(c)

    L, chol_counts = toeplitz_cholesky_schur(c, return_counts=True)
    chol_rel_err = np.linalg.norm(L @ L.conj().T - T) / np.linalg.norm(T)

    rng = np.random.default_rng(1)
    b = rng.normal(size=n) + 1j * rng.normal(size=n)
    x, subst_counts = solve_from_chol_dec(L, b, return_counts=True)
    x_ref = np.linalg.solve(T, b)
    x_rel_err = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
    inv_T, inv_counts = invert_from_chol_dec(L, return_counts=True)
    inv_rel_err = np.linalg.norm(inv_T @ T - np.eye(n)) / np.linalg.norm(np.eye(n))

    print("Cholesky relative error: ", chol_rel_err)
    print("Solution relative_error: ", x_rel_err)
    print("Invertion relative_error: ", inv_rel_err)
    print("n =", n)
    print("Cholesky counts. true: ", chol_counts, " | theory: ", chol_dec_th(n))
    print("Substitutions counts. true: ", subst_counts, " | theory: ", solve_chol_th(n))
    print("Inversion counts. true: ", inv_counts, " | theory: ", inv_chol_th(n))


if __name__ == "__main__":
    demo(100)
