from itertools import combinations

import numpy as np
from scipy.sparse.csgraph import shortest_path  # ZG

# there are many shortest paths - https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.shortest_path.html


def _as_2d_adjacency(bma):
    if hasattr(bma, "adjacencyMatrix"):
        adjacency = np.asarray(bma.adjacencyMatrix, dtype=float)
    elif hasattr(bma, "adjacency_matrix"):
        adjacency = np.asarray(bma.adjacency_matrix, dtype=float)
    else:
        raise AttributeError("BMA object must define `adjacencyMatrix` or `adjacency_matrix`.")

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency matrix must be a square 2D matrix.")
    return adjacency


def _shortest_paths_from_source(adjacency, source):
    return shortest_path(
        adjacency, directed=False, indices=int(source), return_predecessors=True
    )


def _reconstruct_path(pred, source, target):
    """Reconstruct path using predecessors from `source` to `target`."""
    source = int(source)
    target = int(target)
    path = [target]
    node = target
    while node != source:
        node = int(pred[node])
        if node < 0:
            return []
        path.append(node)
    return path


def calculate_branches_for_bma(bma):
    n_points = len(bma.pointsArray)
    bma.branchNumber = np.zeros((n_points, n_points))
    adjacency = _as_2d_adjacency(bma)
    adjacency_sum = np.sum(adjacency, axis=1)  # ZG

    # ZG
    bma.pointType = np.zeros(n_points)
    bma.pointType[adjacency_sum >= 3] = 3
    bma.pointType[adjacency_sum == 1] = 1
    bma.pointType[adjacency_sum == 2] = 0

    # ZG
    triple_adjacents_mask = bma.pointType == 3

    bma.adjacencyMatrix = adjacency

    # ZG
    triple_adjacents = np.where(triple_adjacents_mask)[0]

    # ZG
    bma.pointType[triple_adjacents[bma.pointType[triple_adjacents] == 0]] = 2
    bma.pointType[triple_adjacents[bma.pointType[triple_adjacents] == 1]] = 4
    triplepoint = np.where(np.isin(bma.pointType, [3]))[0]
    singlepoint = np.where(np.isin(bma.pointType, [1]))[0]
    nubpoint = np.where(np.isin(bma.pointType, [2]))[0]

    nubcount = np.ones(len(nubpoint))
    branches = []

    # case 0: single branch (no triple points)
    if len(triplepoint) == 0:
        if len(singlepoint) >= 2:  # ZG
            _, pred = _shortest_paths_from_source(bma.adjacencyMatrix, singlepoint[0])
            bpath = _reconstruct_path(pred, singlepoint[0], singlepoint[1])
            if bpath:
                branches.append(bpath)

    # case 1: single points adjacent to triple points

    # ZG
    adjacency_matrix_2d = bma.adjacencyMatrix
    startpt1, endpt1 = np.where(adjacency_matrix_2d[triplepoint, :][:, singlepoint])

    for branchno in range(len(startpt1)):
        branches.append(
            [triplepoint[startpt1[branchno]], singlepoint[endpt1[branchno]]]
        )  # ZG

    # case 2: triple points adjacent to triple points
    startpt2, endpt2 = np.where(
        adjacency_matrix_2d[triplepoint, :][:, triplepoint]
    )  # ZG
    ind = startpt2 < endpt2
    startpt2, endpt2 = startpt2[ind], endpt2[ind]
    startind = len(branches)
    for branchno in range(startind, len(startpt2) + startind):
        branches.append(
            [
                triplepoint[startpt2[branchno - startind]],
                triplepoint[endpt2[branchno - startind]],
            ]
        )

    if len(triplepoint) > 0:
        # case 3: single points not adjacent to triple points
        predpath = {}
        dists = np.inf * np.ones((len(triplepoint), len(singlepoint)))

        for ll in range(len(singlepoint)):
            dist, pred = _shortest_paths_from_source(
                bma.adjacencyMatrix, singlepoint[ll]
            )  # ZG
            dists[:, ll] = dist[triplepoint]
            predpath[ll] = pred

        badinds = np.where(np.any(np.isinf(dists), axis=0))[0]
        dists = np.delete(dists, badinds, axis=1)
        singlepoint = np.delete(singlepoint, badinds)
        keep_mask = np.ones(len(predpath), dtype=bool)
        keep_mask[badinds] = False
        predpath = [predpath[k] for k in range(len(keep_mask)) if keep_mask[k]]

        # ZG
        if dists.size > 0:
            endpts3 = np.argmin(dists, axis=0)
            branchstarts = triplepoint[endpts3]
            startind = len(branches)

            for branchno in range(startind, len(endpts3) + startind):
                local_idx = branchno - startind
                source = singlepoint[local_idx]
                target = branchstarts[local_idx]
                bpath = _reconstruct_path(predpath[local_idx], source, target)
                if bpath:
                    branches.append(bpath)
                    nubcount[np.isin(nubpoint, bpath)] = 0

        # case 4: triple point to triple point via multiple regular points
        predpath = {}
        nubpoint2 = nubpoint[nubcount.astype(bool)]
        longbranch = np.zeros(len(nubpoint2), dtype=int)
        tempadjacency = bma.adjacencyMatrix.copy()

        tempadjacency[
            np.ix_(np.where(bma.pointType == 3)[0], np.where(bma.pointType == 3)[0])
        ] = 0  # ZG

        # subcase a: triple, regular, triple branch
        for ll in range(len(nubpoint2)):
            if not np.any(tempadjacency[nubpoint2[ll], :]):
                longbranch[ll] = 0
                tripnbrs = np.where(bma.adjacencyMatrix[nubpoint2[ll], :])[0]
                if tripnbrs.size >= 2:
                    branch = [int(tripnbrs[0]), int(nubpoint2[ll]), int(tripnbrs[1])]
                    branches.append(branch)
                    nubcount[np.isin(nubpoint, branch)] = 0
            else:
                longbranch[ll] = 1

        # subcase b: triple, multiple regulars, triple
        if np.any(longbranch):
            nubpoint2 = nubpoint[nubcount.astype(bool)]
            dists = np.inf * np.ones((len(nubpoint2), len(nubpoint2)))

            for ll in range(len(nubpoint2)):
                dist, pred = _shortest_paths_from_source(tempadjacency, nubpoint2[ll])
                dists[:, ll] = dist[nubpoint2]
                predpath[ll] = pred

            dists[dists == 0] = np.inf
            _, endpts4 = np.min(dists, axis=0)
            idx = np.where(np.arange(len(endpts4)) > endpts4)[0]
            branchstarts = nubpoint2[endpts4[idx]]
            branchends = nubpoint2[idx]
            startind = len(branches)

            for branchno in range(startind, len(branchstarts) + startind):
                local_idx = branchno - startind
                end_node = int(branchends[local_idx])
                start_node = int(branchstarts[local_idx])
                triplestart = triplepoint[
                    np.isin(
                        triplepoint,
                        np.where(bma.adjacencyMatrix[end_node, :])[0],
                    )
                ]

                if len(triplestart) == 1:
                    start_triple = triplepoint[
                        np.isin(
                            triplepoint, np.where(bma.adjacencyMatrix[start_node, :])[0]
                        )
                    ]
                    if len(start_triple) != 1:
                        continue

                    middle_path = _reconstruct_path(
                        predpath[idx[local_idx]], end_node, start_node
                    )
                    if not middle_path:
                        continue
                    bpath = [int(start_triple[0])] + middle_path + [int(triplestart[0])]
                elif len(triplestart) == 2:
                    bpath = [
                        int(triplestart[0]),
                        end_node,
                        int(triplestart[1]),
                    ]
                else:
                    continue

                branches.append(bpath)

    # Insert branch point order number into column with index = branchno
    for branchno in range(len(branches)):
        orderind = np.arange(1, len(branches[branchno]) + 1)
        bma.branchNumber[branches[branchno], branchno] = orderind

    # Remove extra columns from bma.branchNumber
    bma.branchNumber = bma.branchNumber[:, np.sum(bma.branchNumber, axis=0) != 0]

    # Create branch adjacency matrix
    bma.branchAdjacency = np.zeros(
        (bma.branchNumber.shape[1], bma.branchNumber.shape[1])
    )

    for ind in range(len(bma.pointsArray)):
        # ZG
        branchind = np.where(bma.branchNumber[ind, :])[0]

        if len(branchind) >= 2:
            adjind = list(combinations(branchind, 2))
            for ind2 in range(len(adjind)):
                bma.branchAdjacency[adjind[ind2][0], adjind[ind2][1]] = 1
                bma.branchAdjacency[adjind[ind2][1], adjind[ind2][0]] = 1
