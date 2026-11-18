"""Plotly helpers for spherical grid visualization."""

from __future__ import annotations

from functools import reduce

import numpy as np
import plotly.graph_objects as go


def tri_indices(simplices):
    """Return Plotly ``i``, ``j``, and ``k`` triangle index lists from simplices."""

    return [[triplet[c] for triplet in simplices] for c in range(3)]


def plotly_trisurf(x, y, z, simplices, face_color="blue", opacity=0.9, plot_edges=None):
    """Build Plotly traces for a triangulated 3D surface."""

    points3D = np.vstack((x, y, z)).T
    # vertices of the surface triangles
    tri_vertices = map(
        lambda index: points3D[index],
        simplices,
    )

    # mean values of z-coordinates of triangle vertices
    # zmean = [
    #     np.mean(tri[:, 2]) for tri in tri_vertices
    # ]
    # # min_zmean = np.min(zmean)
    # max_zmean = np.max(zmean)
    I_inx, J_inx, K_inx = tri_indices(simplices)

    triangles = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=I_inx,
        j=J_inx,
        k=K_inx,
        name="",
        color=face_color,
        opacity=opacity,
    )

    if plot_edges is None:  # the triangle sides are not plotted
        return [triangles]
    else:
        # define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end
        # points for each triangle
        # None separates data corresponding to two consecutive triangles
        lists_coord = [
            [[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices]
            for c in range(3)
        ]
        Xe, Ye, Ze = (reduce(lambda x, y: x + y, lists_coord[k]) for k in range(3))

        # define the lines to be plotted
        lines = go.Scatter3d(
            x=Xe,
            y=Ye,
            z=Ze,
            mode="lines",
            line=dict(color="rgb(50,50,50)", width=1.5),
        )

        return [triangles, lines]
