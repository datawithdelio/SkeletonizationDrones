import numpy as np
from scipy.sparse import csr_matrix


def calculate_medial_order(medial_data):
    """Build medial adjacency order from shared boundary-point triangles."""
    m2 = medial_data[:, 0]
    triin2 = np.column_stack(
        (
            medial_data[:, 2].real.astype(int),
            medial_data[:, 3].real.astype(int),
            medial_data[:, 4].real.astype(int),
        )
    )
    nt = triin2.shape[0]

    # Sparse triangle-incidence matrix (triangle -> boundary-point).
    incidence = csr_matrix(
        (np.ones(3 * nt, dtype=np.int8), (np.repeat(np.arange(nt), 3), triin2.ravel()))
    )

    # Pair triangles that share at least two boundary points.
    shared = incidence @ incidence.T
    shared = shared.tocoo()
    keep = (shared.data > 1) & (shared.row > shared.col)
    a1 = shared.row[keep]
    b1 = shared.col[keep]

    mord = np.vstack((m2[a1], m2[b1]))
    return mord
