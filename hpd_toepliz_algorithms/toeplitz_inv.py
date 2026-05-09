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


def demo(n=8):
    # Positive-definite Hermitian Toeplitz example:
    # c[k] = rho^k * exp(i phi k), |rho| < 1.
    rho = 0.72
    phi = 0.37
    c = np.array(
        [rho**k * np.exp(1j * phi * k) for k in range(n)],
        dtype=np.complex128,
    )

    T = hermitian_toeplitz(c)

    L, chol_counts = toeplitz_cholesky_schur(c, return_counts=True)
    chol_rel_err = np.linalg.norm(L @ L.conj().T - T) / np.linalg.norm(T)

    print("n =", n)
    print("Cholesky relative error:", chol_rel_err)
    print("Cholesky counts:", chol_counts)


if __name__ == "__main__":
    demo(100)
