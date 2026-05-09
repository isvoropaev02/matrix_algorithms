import numpy as np
from utilities import OpCount, hermitian_toeplitz


def levinson_hermitian_solve(c, b, return_counts=False):
    """Levinson-Durbin solver for Hermitian positive-definite Toeplitz T x = b."""
    c = np.asarray(c, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    n = len(b)
    if len(c) < n:
        raise ValueError("len(c) must be >= len(b)")
    if n == 0:
        x = np.empty(0, dtype=np.complex128)
        return (x, OpCount()) if return_counts else x
    if abs(c[0].imag) > 1e-13 or c[0].real <= 0:
        raise ValueError("c[0] must be positive real for HPD Hermitian Toeplitz.")
    cnt = OpCount()
    a = np.empty(0, dtype=np.complex128)
    E = c[0].real
    x = np.zeros(n, dtype=np.complex128)
    x[0] = b[0] / E
    cnt.rdiv += 2
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
        row_dot = c[m] * x[0]
        cnt.rmul += 4
        cnt.radd += 2
        for j in range(1, m):
            row_dot = row_dot + c[m - j] * x[j]
            cnt.rmul += 4
            cnt.radd += 4
        gamma = (b[m] - row_dot) / E
        cnt.radd += 2
        cnt.rdiv += 2
        old_x = x[:m].copy()
        for j in range(m):
            x[j] = old_x[j] + gamma * np.conj(a[m - 1 - j])
            cnt.rmul += 4
            cnt.radd += 4
        x[m] = gamma
    return (x, cnt) if return_counts else x


def levinson_complexity_th(n=8):
    return OpCount(rmul=(8 * n**2 - 13 * n + 5), radd=(8 * n**2 - 14 * n + 6), rdiv=4 * n - 2, rsqrt=0)


def demo(n=8):
    # Positive-definite Hermitian Toeplitz example
    rho = 0.90
    phi = 0.37
    c = np.array(
        [rho**k * np.exp(1j * phi * k) for k in range(n)],
        dtype=np.complex128,
    )
    c[0] += 0.001
    T = hermitian_toeplitz(c)

    print("cond: ", np.linalg.cond(T))

    # Levinson solve for a random right-hand side
    rng = np.random.default_rng(123)
    b = rng.normal(size=n) + 1j * rng.normal(size=n)
    x, lev_counts = levinson_hermitian_solve(c, b, return_counts=True)
    solve_rel_err = np.linalg.norm(T @ x - b) / np.linalg.norm(b)
    x_ref = np.linalg.solve(T, b)
    lev_ref_err = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)

    print(f"n = {n}")
    print(f"Levinson solve rel. residual: {solve_rel_err:.2e}")
    print(f"Levinson vs np.solve rel.err:{lev_ref_err:.2e}\n")
    print("Operation counts:")
    print(f"  Levinson (true):   {lev_counts}")
    print(f"  Levinson (theory): {levinson_complexity_th(n)}")


if __name__ == "__main__":
    demo(1200)
