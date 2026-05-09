import numpy as np
from utilities import OpCount, hermitian_toeplitz


def trench_inverse(c, return_counts=False):
    """
    Trench algorithm for inverse of Hermitian positive-definite Toeplitz matrix.
    Returns the full inverse matrix T^{-1} in O(n^2) operations.
    """
    c = np.asarray(c, dtype=np.complex128)
    n = len(c)
    if n == 0:
        T_inv = np.empty((0, 0), dtype=np.complex128)
        return (T_inv, OpCount()) if return_counts else T_inv
    if abs(c[0].imag) > 1e-13 or c[0].real <= 0:
        raise ValueError("c[0] must be positive real for HPD Hermitian Toeplitz.")
    cnt = OpCount()
    # Levinson recursion for prediction coefficients a and final error E
    a = np.empty(0, dtype=np.complex128)
    E = c[0].real
    for m in range(1, n):
        delta = c[m]
        for j in range(1, m):
            delta = delta + a[j - 1] * c[m - j]
            cnt.rmul += 4
            cnt.radd += 4
        kappa = -delta / E
        cnt.rdiv += 2
        old_a = a.copy()
        a_new = np.empty(m, dtype=np.complex128)
        for j in range(1, m):
            a_new[j - 1] = old_a[j - 1] + kappa * np.conj(old_a[m - j - 1])
            cnt.rmul += 4
            cnt.radd += 4
        a_new[m - 1] = kappa
        a = a_new
        E = E * (1.0 - (kappa.real * kappa.real + kappa.imag * kappa.imag))
        cnt.rmul += 3
        cnt.radd += 2
        if E <= 0:
            raise np.linalg.LinAlgError("Toeplitz matrix is not positive definite.")
    # First column p of the inverse
    p = np.empty(n, dtype=np.complex128)
    p[0] = 1.0 / E
    cnt.rdiv += 1
    for k in range(1, n):
        p[k] = a[k - 1] / E
        cnt.rdiv += 2  # complex divided by real
    # Fill the inverse matrix using Trench recurrence
    T_inv = np.empty((n, n), dtype=np.complex128)
    T_inv[0, 0] = p[0].real  # guaranteed real because p[0] = 1/E > 0
    for j in range(1, n):
        T_inv[0, j] = np.conj(p[j])
        T_inv[j, 0] = p[j]
    # Recurrence: T_inv[i,j] = T_inv[i-1,j-1] + (p[i]*conj(p[j]) - conj(p[n-i])*p[n-j]) / p[0]
    p0 = p[0]
    for i in range(1, n):
        # we can precompute conj(p[n-i]) to save operations
        conj_p_ni = np.conj(p[n - i])
        for j in range(1, n):
            term = p[i] * np.conj(p[j]) - conj_p_ni * p[n - j]
            cnt.rmul += 8
            cnt.radd += 6
            T_inv[i, j] = T_inv[i - 1, j - 1] + term / p0
            cnt.rdiv += 2  # division of complex term by real p0
    # No need to set upper-left corners – already set
    return (T_inv, cnt) if return_counts else T_inv


def trench_complexity_th(n=8):
    return OpCount(rmul=(12 * n**2 - 25 * n + 13), radd=(10 * n**2 - 22 * n + 12), rdiv=2 * n**2 - 1, rsqrt=0)


def demo(n=8):
    # Positive-definite Hermitian Toeplitz example
    rho = 0.72
    phi = 0.37
    c = np.array(
        [rho**k * np.exp(1j * phi * k) for k in range(n)],
        dtype=np.complex128,
    )
    T = hermitian_toeplitz(c)

    # Trench inverse
    T_inv_trench, trench_counts = trench_inverse(c, return_counts=True)
    T_inv_exact = np.linalg.inv(T)
    trench_rel_err = np.linalg.norm(T_inv_trench - T_inv_exact) / np.linalg.norm(T_inv_exact)

    check_e = T @ T_inv_trench
    check_e[np.abs(check_e) < 1e-15] = 0
    print(check_e)

    print(f"n = {n}")
    print(f"Trench inverse rel. error:    {trench_rel_err:.2e}")
    print("Operation counts:")
    print(f"  Trench (true):   {trench_counts}")
    print(f"  Trench (theory): {trench_complexity_th(n)}")


if __name__ == "__main__":
    demo(3)
