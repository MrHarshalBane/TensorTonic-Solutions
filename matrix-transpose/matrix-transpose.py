import numpy as np


def matrix_transpose(A):
    n = len(A)
    m = len(A[0])
    result = np.empty(
        (m, n), dtype=np.result_type(*[np.asarray(row) for row in A])
    )
    for i in range(n):
        for j in range(m):
            result[j, i] = A[i][j]
    return result
