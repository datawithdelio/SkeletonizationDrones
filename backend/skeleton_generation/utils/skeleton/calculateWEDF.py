import copy

import numpy as np


def _tri_area(c1, c2, c3):
    """Triangle area from complex boundary points."""
    return abs(np.imag((c3 - c1) * np.conj(c2 - c1))) / 2.0


def _point_type_array(obj):
    if hasattr(obj, "pointType"):
        return np.asarray(obj.pointType)
    if hasattr(obj, "point_type"):
        return np.asarray(obj.point_type)
    raise AttributeError("BMA object must define `pointType` or `point_type`.")


def calculate_wedf(bma):
    temp = copy.deepcopy(bma)
    temp.WEDFArray = np.inf * np.ones(len(temp.pointsArray))
    bma.WEDFArray = np.inf * np.ones(len(temp.pointsArray))

    indices_of_constrained_ends = temp.find_constrained_ends()

    indice_pts_boundary = temp.indexOfBndryPoints[indices_of_constrained_ends]
    num_indices_oce = len(indices_of_constrained_ends)

    for i in range(num_indices_oce):
        pt1 = temp.boundary[indice_pts_boundary[i, 0]]
        pt2 = temp.boundary[indice_pts_boundary[i, 1]]
        pt3 = temp.boundary[indice_pts_boundary[i, 2]]
        temp.WEDFArray[indices_of_constrained_ends[i]] = _tri_area(pt1, pt2, pt3)

    bma.WEDFArray[indices_of_constrained_ends] = temp.WEDFArray[
        indices_of_constrained_ends
    ]

    if not np.any(np.isfinite(temp.WEDFArray)):
        bma.onMedialResidue = np.zeros(len(bma.pointsArray), dtype=bool)
        return bma

    smallest = np.min(temp.WEDFArray)
    index_of_smallest = np.argmin(temp.WEDFArray)

    end_loop = False
    point_type = _point_type_array(temp)

    while not end_loop:
        parent_indices = np.where(temp.adjacencyMatrix[index_of_smallest])[0]
        assert (
            len(parent_indices) == 1
        ), f"Zero or more than one parent at index {index_of_smallest}. Make sure your graph is connected."
        index_of_parent = int(parent_indices[0])

        # remove_at_index mutates in place in this codebase.
        temp.remove_at_index(index_of_smallest)
        point_type = _point_type_array(temp)

        if index_of_smallest < index_of_parent:
            index_of_parent -= 1

        if len(np.where(temp.adjacencyMatrix[index_of_parent])[0]) == 1:
            pt1 = temp.boundary[temp.indexOfBndryPoints[index_of_parent, 0]]
            pt2 = temp.boundary[temp.indexOfBndryPoints[index_of_parent, 1]]
            pt3 = temp.boundary[temp.indexOfBndryPoints[index_of_parent, 2]]

            if point_type[index_of_parent] != 3:
                temp.WEDFArray[index_of_parent] = smallest + _tri_area(pt1, pt2, pt3)
            else:
                nubinds = np.where(
                    bma.adjacencyMatrix[
                        np.where(
                            np.isin(bma.pointsArray, temp.pointsArray[index_of_parent])
                        )
                    ]
                )[0]
                nubinds = nubinds[np.isinf(bma.WEDFArray[nubinds]) == False]
                nub_vals = np.sum(bma.WEDFArray[nubinds])
                temp.WEDFArray[index_of_parent] = _tri_area(pt1, pt2, pt3) + nub_vals

            bma.WEDFArray[
                bma.pointsArray == temp.pointsArray[index_of_parent]
            ] = temp.WEDFArray[index_of_parent]

        if not np.any(np.isfinite(temp.WEDFArray)):
            bma.onMedialResidue = np.zeros(len(bma.pointsArray), dtype=bool)
            break

        smallest = np.min(temp.WEDFArray)
        index_of_smallest = np.argmin(temp.WEDFArray)

        if len(temp.pointsArray) == 1 or not temp.find_constrained_ends():
            bma.onMedialResidue = np.zeros(len(bma.pointsArray), dtype=bool)
            bma.onMedialResidue = np.isin(bma.pointsArray, temp.pointsArray)
            end_loop = True

    return bma
