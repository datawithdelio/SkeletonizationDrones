import numpy as np


def _validate_triplets(rows, cols, vals):
    if rows.shape != cols.shape:
        raise ValueError(
            f"length of row indices ({rows.size}) not equal to column indices ({cols.size})"
        )
    if vals is not None and vals.shape != rows.shape:
        raise ValueError(
            f"length of values ({vals.size}) not equal to row indices ({rows.size})"
        )
    if rows.size > 0 and (rows.min() < 0 or cols.min() < 0):
        raise ValueError("row and column indices must be >= 0")


def sparse_to_csr(A, *args):
    """
    Build CSR arrays from either a sparse matrix or triplet indices.

    Usage:
      - sparse_to_csr(sparse_matrix) -> (rp, ci, ai)
      - sparse_to_csr(nzi, nzj[, nzv[, nrow[, ncol]]]) -> rp or (rp, ci, ai, ncol)
    """
    if len(args) == 0:
        # Sparse matrix input.
        nrow, ncol = A.shape
        coo = A.tocoo()
        nzi = coo.row.astype(np.int64, copy=False)
        nzj = coo.col.astype(np.int64, copy=False)
        nzv = coo.data.astype(float, copy=False)

        order = np.lexsort((nzj, nzi))
        nzi = nzi[order]
        nzj = nzj[order]
        nzv = nzv[order]

        rp = np.zeros(nrow + 1, dtype=np.int64)
        np.add.at(rp, nzi + 1, 1)
        np.cumsum(rp, out=rp)
        return rp, nzj, nzv

    # Triplet input.
    nzi = np.asarray(A, dtype=np.int64).reshape(-1)
    nzj = np.asarray(args[0], dtype=np.int64).reshape(-1)
    nzv = None

    has_values = len(args) >= 2 and args[1] is not None and np.asarray(args[1]).size > 0
    if has_values:
        nzv = np.asarray(args[1], dtype=float).reshape(-1)

    _validate_triplets(nzi, nzj, nzv)

    if len(args) >= 3:
        nrow = int(args[2])
    else:
        nrow = int(nzi.max() + 1) if nzi.size > 0 else 0

    if len(args) >= 4:
        ncol = int(args[3])
    else:
        ncol = int(nzj.max() + 1) if nzj.size > 0 else 0

    if nrow < 0 or ncol < 0:
        raise ValueError("nrow and ncol must be non-negative")
    if nzi.size > 0 and (nzi.max() >= nrow or nzj.max() >= ncol):
        raise ValueError("triplet indices exceed provided matrix dimensions")

    order = np.lexsort((nzj, nzi))
    nzi = nzi[order]
    nzj = nzj[order]
    if nzv is not None:
        nzv = nzv[order]

    rp = np.zeros(nrow + 1, dtype=np.int64)
    np.add.at(rp, nzi + 1, 1)
    np.cumsum(rp, out=rp)

    if nzv is None:
        return rp
    return rp, nzj, nzv, ncol
