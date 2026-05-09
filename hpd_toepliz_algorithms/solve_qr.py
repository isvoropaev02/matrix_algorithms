import numpy as np
from utilities import OpCount, hermitian_toeplitz


def solve_qr(c, b, return_counts=False):
    """
    Solve T x = b for Hermitian positive-definite Toeplitz T via fast QR.
    Uses O(n^2) operations.
    """
    c = np.asarray(c, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    n = len(b)
    if len(c) < n:
        raise ValueError("len(c) must be >= len(b)")
    if n == 0:
        return (np.empty(0, dtype=np.complex128), OpCount()) if return_counts else np.empty(0, dtype=np.complex128)
    if abs(c[0].imag) > 1e-13 or c[0].real <= 0:
        raise ValueError("c[0] must be positive real for HPD Hermitian Toeplitz.")

    cnt = OpCount()
    # Normalize generators
    s0 = np.sqrt(c[0].real)
    cnt.rsqrt += 1
    # g0, g1 are Schur generators for T
    g0 = c.astype(np.complex128) / s0
    g1 = g0.copy()
    cnt.rdiv += 2 * n  # division of real s0

    # R will be stored compactly; we need the diagonal and superdiagonal of R
    # We'll accumulate R in a 2D array of size n x n, but fill only upper triangle
    R = np.zeros((n, n), dtype=np.complex128)
    # We also need to apply rotations to b
    xb = b.astype(np.complex128).copy()

    # First column -> first row of R
    for j in range(n):
        R[0, j] = g0[j]
    # shift g0
    for j in range(n - 1, 0, -1):
        g0[j] = g0[j - 1]
    g0[0] = 0.0

    for i in range(1, n):
        # determine rotation to zero g1[i] while keeping g0[i] real
        denom = g0[i].real
        # angle such that new g0[i] becomes sqrt(|g0[i]|^2 + |g1[i]|^2) (real)
        rho = -g1[i] / denom
        cnt.rdiv += 2
        # gamma = sqrt(1 + |rho|^2) because we want to normalize
        gamma = np.sqrt(1.0 + (rho.real * rho.real + rho.imag * rho.imag))
        cnt.rsqrt += 1
        # apply rotation to columns i..n-1 of both generators and to b
        for j in range(i, n):
            alpha = g0[j]
            beta = g1[j]
            new_g0 = (alpha - np.conj(rho) * beta) / gamma
            new_g1 = (rho * alpha + beta) / gamma
            cnt.rmul += 8
            cnt.radd += 8
            cnt.rdiv += 4
            g0[j] = new_g0
            g1[j] = new_g1

        # apply rotation to b (only the two relevant components)
        # The rotation matrix is [c s; -conj(s) c] with c=1/gamma, s=-conj(rho)/gamma
        # We apply it to b[i-1] and b[i] (but careful: index shift)
        # Actually we store the rotations and apply them in the correct order.
        # For brevity, we apply directly:
        c_val = 1.0 / gamma
        s_val = -np.conj(rho) / gamma
        # update b[i-1] and b[i]
        bi_1 = xb[i - 1]
        bi = xb[i]
        xb[i - 1] = c_val * bi_1 + s_val * bi  # This is Q^H * b
        xb[i] = -np.conj(s_val) * bi_1 + c_val * bi
        cnt.rmul += 16
        cnt.radd += 12
        cnt.rdiv += 3

        # Now g0[i] is real positive and we store it as R's diagonal
        for j in range(i, n):
            R[i, j] = g0[j]

        # shift g0
        for j in range(n - 1, i, -1):
            g0[j] = g0[j - 1]
        g0[i] = 0.0

    # After the loop, R is upper triangular. Solve R x = xb by back-substitution.
    x = np.zeros(n, dtype=np.complex128)
    for i in range(n - 1, -1, -1):
        s = xb[i]
        for j in range(i + 1, n):
            s -= R[i, j] * x[j]
            cnt.rmul += 4
            cnt.radd += 4
        x[i] = s / R[i, i]
        cnt.rdiv += 2
    return (x, cnt) if return_counts else x


def demo(n=8):
    # Positive-definite Hermitian Toeplitz example
    rho = 0.80
    phi = 0.37
    c = np.array(
        [rho**k * np.exp(1j * phi * k) for k in range(n)],
        dtype=np.complex128,
    )
    # c[0] += 0.001
    T = hermitian_toeplitz(c)

    print("cond: ", np.linalg.cond(T))

    rng = np.random.default_rng(123)
    b = rng.normal(size=n) + 1j * rng.normal(size=n)
    x, qr_counts = solve_qr(c, b, return_counts=True)
    solve_rel_err = np.linalg.norm(T @ x - b) / np.linalg.norm(b)
    x_ref = np.linalg.solve(T, b)
    x_rel_err = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)

    print(f"n = {n}")
    print(f"QR solve rel. residual: {solve_rel_err:.2e}")
    print(f"QR vs np.solve rel.err:{x_rel_err:.2e}\n")
    print("Operation counts:")
    print(f"  QR (true):   {qr_counts}")
    # print(f"  Levinson (theory): {levinson_complexity_th(n)}")


if __name__ == "__main__":
    demo(3)
