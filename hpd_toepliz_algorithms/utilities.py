from dataclasses import dataclass
import numpy as np


@dataclass
class OpCount:
    rmul: int = 0
    radd: int = 0
    rdiv: int = 0
    rsqrt: int = 0


def hermitian_toeplitz(c):
    """
    Build Hermitian Toeplitz T from its first column c:
        T[i,j] = c[i-j]              if i >= j
               = conj(c[j-i])        if i < j
    """
    c = np.asarray(c, dtype=np.complex128)
    n = len(c)
    T = np.empty((n, n), dtype=np.complex128)

    for i in range(n):
        for j in range(n):
            if i >= j:
                T[i, j] = c[i - j]
            else:
                T[i, j] = np.conj(c[j - i])

    return T
