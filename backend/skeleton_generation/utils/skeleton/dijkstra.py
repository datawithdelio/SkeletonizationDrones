import heapq

import numpy as np

from utils.skeleton.sparse_to_csr import sparse_to_csr


def _normalize_csr(rp, ci, ai):
    """Normalize CSR arrays to 0-based indexing."""
    rp = np.asarray(rp, dtype=np.int64).reshape(-1)
    ci = np.asarray(ci, dtype=np.int64).reshape(-1)
    ai = np.asarray(ai, dtype=float).reshape(-1)

    if rp.size < 2:
        raise ValueError("CSR row pointer `rp` must have length >= 2.")

    if rp[0] == 1:
        rp = rp - 1
    if ci.size > 0 and ci.min() >= 1:
        ci = ci - 1

    if ai.size == ci.size + 1 and rp[-1] == ai.size:
        ai = ai[1:]
    if ci.size == ai.size + 1 and rp[-1] == ci.size:
        ci = ci[1:]

    if ci.size != ai.size:
        raise ValueError("CSR columns (`ci`) and weights (`ai`) must have the same length.")
    if rp[-1] != ci.size:
        raise ValueError("CSR row pointer does not match edge array lengths.")
    if np.any(rp[1:] < rp[:-1]):
        raise ValueError("CSR row pointer (`rp`) must be non-decreasing.")
    if ci.size > 0 and (ci.min() < 0 or ci.max() >= rp.size - 1):
        raise ValueError("CSR column indices are out of bounds.")

    return rp, ci, ai


def dijkstra(A, u):
    """
    Compute shortest-path distances from source node `u` using Dijkstra.

    Args:
        A: Graph in CSR form as dict with keys {rp, ci, ai} or a sparse matrix.
        u: Source node index (0-based).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - dist: shortest distances from `u` to every node.
            - pred: predecessor index for each node (-1 for unreachable nodes).
    """
    if isinstance(A, dict):
        missing = {"rp", "ci", "ai"} - set(A)
        if missing:
            raise ValueError(f"Missing CSR keys: {sorted(missing)}")
        rp, ci, ai = _normalize_csr(A["rp"], A["ci"], A["ai"])
    else:
        rp, ci, ai = sparse_to_csr(A)
        rp, ci, ai = _normalize_csr(rp, ci, ai)

    n = rp.size - 1
    if not (0 <= u < n):
        raise ValueError(f"Source node `u` must be in [0, {n - 1}], got {u}.")
    if np.any(ai < 0):
        raise ValueError("Dijkstra's algorithm cannot handle negative edge weights.")

    dist = np.full(n, np.inf, dtype=float)
    pred = np.full(n, -1, dtype=np.int64)
    dist[u] = 0.0
    pred[u] = u

    heap = [(0.0, int(u))]
    while heap:
        current_dist, node = heapq.heappop(heap)
        if current_dist != dist[node]:
            continue

        row_start = rp[node]
        row_end = rp[node + 1]
        for edge_idx in range(row_start, row_end):
            neighbor = int(ci[edge_idx])
            candidate_dist = current_dist + ai[edge_idx]
            if candidate_dist < dist[neighbor]:
                dist[neighbor] = candidate_dist
                pred[neighbor] = node
                heapq.heappush(heap, (candidate_dist, neighbor))

    return dist, pred
