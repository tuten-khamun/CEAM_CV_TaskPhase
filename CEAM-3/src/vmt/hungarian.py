"""Exact Kuhn-Munkres (Hungarian) algorithm implementation in pure NumPy and Python.

Provides linear_sum_assignment(cost_matrix) with the exact same interface and semantics
as scipy.optimize.linear_sum_assignment, ensuring zero external dependencies.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def linear_sum_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the linear sum assignment problem (minimum weight bipartite matching).

    Finds a 1-to-1 matching between rows and columns of cost_matrix that minimizes
    the total assignment cost using the Hungarian (Kuhn-Munkres) dual-potential method.

    Args:
        cost_matrix: 2D array of shape (N, M).

    Returns:
        Tuple of (row_ind, col_ind) arrays containing optimal assignment pairs.
    """
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.size == 0 or cost.shape[0] == 0 or cost.shape[1] == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    transposed = False
    if cost.shape[0] > cost.shape[1]:
        cost = cost.T
        transposed = True

    n_rows, n_cols = cost.shape
    u = np.zeros(n_rows + 1)
    v = np.zeros(n_cols + 1)
    p = np.zeros(n_cols + 1, dtype=int)
    way = np.zeros(n_cols + 1, dtype=int)

    for i in range(1, n_rows + 1):
        p[0] = i
        minv = np.full(n_cols + 1, np.inf)
        used = np.zeros(n_cols + 1, dtype=bool)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n_cols + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    ans = np.zeros(n_rows, dtype=int)
    for j in range(1, n_cols + 1):
        if 0 < p[j] <= n_rows:
            ans[p[j] - 1] = j - 1

    row_ind = np.arange(n_rows)
    col_ind = ans

    if transposed:
        order = np.argsort(col_ind)
        return col_ind[order], row_ind[order]
    return row_ind, col_ind
