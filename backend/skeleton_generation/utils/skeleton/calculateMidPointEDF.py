import numpy as np


def _get_attr(obj, *names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name), name
    raise AttributeError(f"Object is missing expected attributes: {names}")


def _set_if_exists(obj, value, *names):
    for name in names:
        if hasattr(obj, name):
            setattr(obj, name, value)


def calculate_mid_point_edf(bma):
    temp = bma.copy()
    index_bndry_temp, _ = _get_attr(temp, "index_of_bndry_points", "indexOfBndryPoints")
    edge_points_temp, _ = _get_attr(temp, "edge_points_array", "edgePointsArray")
    radii_temp, _ = _get_attr(temp, "radii_array", "radiiArray")
    edf_temp, _ = _get_attr(temp, "edf_array", "EDFArray")
    adjacency_temp, _ = _get_attr(temp, "adjacency_matrix", "adjacencyMatrix")
    points_temp, _ = _get_attr(temp, "points_array", "pointsArray")

    index_bndry_bma, _ = _get_attr(bma, "index_of_bndry_points", "indexOfBndryPoints")
    edge_points_bma, _ = _get_attr(bma, "edge_points_array", "edgePointsArray")
    radii_bma, _ = _get_attr(bma, "radii_array", "radiiArray")
    edf_bma, _ = _get_attr(bma, "edf_array", "EDFArray")
    points_bma, _ = _get_attr(bma, "points_array", "pointsArray")

    indices_of_constrained_ends = temp.find_constrained_ends()
    if isinstance(indices_of_constrained_ends, tuple):
        indices_of_constrained_ends = indices_of_constrained_ends[0]
    indice_pts_boundary = index_bndry_temp[indices_of_constrained_ends]
    num_indices_oce = len(indices_of_constrained_ends)

    for i in range(num_indices_oce):
        ind1, ind2, ind3 = indice_pts_boundary[i]

        if (ind2 - ind1) * (ind3 - ind2) == 1:
            pass
        else:
            ind = sorted([ind1, ind2, ind3])
            if ind[0] == 1:
                if ind[1] == 2:
                    ind1, ind2, ind3 = ind[2], ind[0], ind[1]
                else:
                    ind1, ind2, ind3 = ind[1], ind[2], ind[0]
            else:
                ind1, ind2, ind3 = ind

            indice_pts_boundary[i] = [ind1, ind2, ind3]

        pt1 = temp.boundary[ind1]
        pt2 = temp.boundary[ind2]
        pt3 = temp.boundary[ind3]
        mid_point = (pt1 + pt3) / 2

        index_bndry_temp[indices_of_constrained_ends[i]] = indice_pts_boundary[i]
        edge_points_temp[indices_of_constrained_ends[i]] = mid_point
        radii_temp[indices_of_constrained_ends[i]] = np.linalg.norm(
            mid_point - pt1
        )
        edf_temp[indices_of_constrained_ends[i]] = np.linalg.norm(mid_point - pt2)

    index_bndry_bma[indices_of_constrained_ends] = index_bndry_temp[indices_of_constrained_ends]
    edge_points_bma[indices_of_constrained_ends] = edge_points_temp[indices_of_constrained_ends]
    radii_bma[indices_of_constrained_ends] = radii_temp[indices_of_constrained_ends]
    edf_bma[indices_of_constrained_ends] = edf_temp[indices_of_constrained_ends]

    if not np.any(np.isfinite(edf_temp)):
        _set_if_exists(bma, edge_points_bma, "points_array", "pointsArray")
        return bma

    smallest, index_of_smallest = np.min(edf_temp), np.argmin(edf_temp)
    end_loop = False

    while not end_loop:
        ind_s1 = index_bndry_temp[index_of_smallest, 0]
        ind_s3 = index_bndry_temp[index_of_smallest, 2]
        edge_point = edge_points_temp[index_of_smallest]
        assert np.allclose(edge_point, (temp.boundary[ind_s1] + temp.boundary[ind_s3]) / 2)

        index_of_parent_arr = np.where(adjacency_temp[index_of_smallest])[0]
        assert (
            len(index_of_parent_arr) == 1
        ), f"Zero or more than one parent at index {index_of_smallest}. Make sure your graph is connected."
        index_of_parent = int(index_of_parent_arr[0])

        temp.remove_at_index(index_of_smallest)
        index_bndry_temp, _ = _get_attr(temp, "index_of_bndry_points", "indexOfBndryPoints")
        edge_points_temp, _ = _get_attr(temp, "edge_points_array", "edgePointsArray")
        radii_temp, _ = _get_attr(temp, "radii_array", "radiiArray")
        edf_temp, _ = _get_attr(temp, "edf_array", "EDFArray")
        adjacency_temp, _ = _get_attr(temp, "adjacency_matrix", "adjacencyMatrix")
        points_temp, _ = _get_attr(temp, "points_array", "pointsArray")

        if index_of_smallest < index_of_parent:
            index_of_parent -= 1

        ind_of_gd_parent = np.where(adjacency_temp[index_of_parent])[0]
        if len(ind_of_gd_parent) == 1:
            ind_of_gd_parent = int(ind_of_gd_parent[0])
            ind_p = index_bndry_temp[index_of_parent]
            o_p1 = np.where(ind_p == ind_s1)[0]
            o_p3 = np.where(ind_p == ind_s3)[0]

            if len(o_p1) == 1 and len(o_p3) == 1:
                ind_p = np.delete(ind_p, [o_p1[0], o_p3[0]])
                ind_p1 = int(ind_p[0])

                ind_gd_pt = index_bndry_temp[ind_of_gd_parent]
                o_gp1 = np.where(ind_gd_pt == ind_p1)[0]

                assert len(o_gp1) == 1

                o_gp3 = np.concatenate(
                    (np.where(ind_gd_pt == ind_s1)[0], np.where(ind_gd_pt == ind_s3)[0])
                )
                assert len(o_gp3) == 1

                ind_p3 = int(ind_gd_pt[o_gp3[0]])

                if ind_p3 == ind_s3:
                    ind_p2 = ind_s1
                elif ind_p3 == ind_s1:
                    ind_p2 = ind_s3
                else:
                    raise RuntimeError("Unexpected midpoint parent reconstruction state.")

                index_bndry_temp[index_of_parent] = [ind_p1, ind_p2, ind_p3]
            else:
                raise RuntimeError(
                    "Could not locate smallest triangle segment endpoints in parent triangle."
                )

            edge_points_temp[index_of_parent] = (
                temp.boundary[ind_p1] + temp.boundary[ind_p3]
            ) / 2
            edf_temp[index_of_parent] = smallest + np.linalg.norm(
                edge_point - edge_points_temp[index_of_parent]
            )
            radii_temp[index_of_parent] = np.linalg.norm(
                edge_points_temp[index_of_parent] - temp.boundary[ind_p1]
            )

            locind = np.where(points_bma == points_temp[index_of_parent])[0]

            index_bndry_bma[locind] = index_bndry_temp[index_of_parent]
            edge_points_bma[locind] = edge_points_temp[index_of_parent]
            edf_bma[locind] = edf_temp[index_of_parent]
            radii_bma[locind] = radii_temp[index_of_parent]

            if not np.any(np.isfinite(edf_temp)):
                end_loop = True
            else:
                smallest, index_of_smallest = np.min(edf_temp), np.argmin(edf_temp)
            if len(points_temp) == 1 or not temp.find_constrained_ends():
                end_loop = True

    _set_if_exists(bma, edge_points_bma, "points_array", "pointsArray")
    return bma
