from __future__ import annotations

import numpy as np

"""Small coordinate and geometry helpers used across :mod:`mars_currents`."""


__author__ = "Apostolos Kolokotronis, David Andrews"
__email__ = "apostolos.kolokotronis@irf.se, david.andrews@irf.se"
__copyright__ = "Copyright 2020-2024"
__license__ = "MIT"
__version__ = "1.0.0"
__status__ = "Prototype"

__all__ = [
    "Rot",
    "bow_shock",
    "cart_to_sph_rotation_mat",
    "cart_to_sph_vf",
    "sph_to_cart_vf",
    "angle_difference",
    "cartesian_to_spherical",
    "spherical_to_cartesian",
    "spherical_unit_vectors",
    "cartesian_vector_to_spherical",
    "spherical_vector_to_cartesian",
]


EPS = 1e-15


def bow_shock(x0, L, eps, th):
    """Return the conic-section bow-shock radius for angle ``th``.

    Parameters
    ----------
    x0 : float
        X-axis offset of the conic section. The current expression does not use
        this value directly, but it is kept for compatibility with existing
        calling code.
    L : float
        Semi-latus rectum of the conic section.
    eps : float
        Eccentricity of the conic section.
    th : float or numpy.ndarray
        Angle in radians.
    """

    return L * (1 + eps * np.cos(th)) ** (-1)


def angle_difference(x, y, degrees=False):
    """Signed smallest distance between two angles x and y.
    Input in radians, or degrees if specified."""
    if degrees:
        conv = np.pi / 180.0
        return np.arctan2(np.sin((x - y) * conv), np.cos((x - y) * conv)) / conv
    return np.arctan2(np.sin(x - y), np.cos(x - y))


# ============================================================
# Coordinate transforms
# ============================================================


def cartesian_to_spherical(x, y, z):
    """
    Cartesian -> spherical coordinates.

    Parameters
    ----------
    x, y, z : array_like

    Returns
    -------
    r, theta, phi : ndarray
        theta = polar angle from +z
        phi   = azimuth from +x
    """

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    r = np.sqrt(x**2 + y**2 + z**2)

    theta = np.arccos(np.clip(z / (r + EPS), -1.0, 1.0))

    phi = np.mod(np.arctan2(y, x), 2 * np.pi)

    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    """
    Spherical -> Cartesian coordinates.
    """

    r = np.asarray(r)
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z


# def cart_to_sph(theta, phi):
def spherical_unit_vectors(theta, phi):
    """
    Returns spherical basis vectors in Cartesian coordinates.

    Output shapes:
        (..., 3)

    Returns
    -------
    e_r, e_theta, e_phi
    """

    theta = np.asarray(theta)
    phi = np.asarray(phi)

    st = np.sin(theta)
    ct = np.cos(theta)

    sp = np.sin(phi)
    cp = np.cos(phi)

    e_r = np.stack([st * cp, st * sp, ct], axis=-1)

    e_theta = np.stack([ct * cp, ct * sp, -st], axis=-1)

    e_phi = np.stack([-sp, cp, np.zeros_like(phi)], axis=-1)

    return e_r, e_theta, e_phi


def cart_to_sph_rotation_mat(theta, phi):
    """Return the rotation matrix to rotate Cartesian vectors into spherical
    components at the given angles."""

    e_r, e_theta, e_phi = spherical_unit_vectors(theta, phi)

    rot_m = np.stack([e_r, e_theta, e_phi], axis=-2)

    return rot_m


# ============================================================
# Vector component transforms
# ============================================================


def cartesian_vector_to_spherical(Vx, Vy, Vz, theta, phi):
    """
    Cartesian vector components -> spherical components.

    Returns
    -------
    Vr, Vtheta, Vphi
    """

    V = np.stack([Vx, Vy, Vz], axis=-1)

    e_r, e_theta, e_phi = spherical_unit_vectors(theta, phi)

    Vr = np.sum(V * e_r, axis=-1)
    Vtheta = np.sum(V * e_theta, axis=-1)
    Vphi = np.sum(V * e_phi, axis=-1)

    return Vr, Vtheta, Vphi


def cart_to_sph_vf(A, th, phi):
    """Wrapper for cartesian_vector_to_spherical."""

    return cartesian_vector_to_spherical(A[..., 0], A[..., 1], A[..., 2], th, phi)


# OLD DEFINITION
# def cart_to_sph_vf(A, th, phi):
#     """Rotate Cartesian vector-field samples into spherical components."""

#     points_rot = [cart_to_sph(theta, phii) for theta, phii in zip(th, phi)]
#     rotated_to_sph = [np.dot(rot_m, point) for rot_m, point in zip(points_rot, A)]

#     return np.array(rotated_to_sph)


def spherical_vector_to_cartesian(Vr, Vtheta, Vphi, theta, phi):
    """
    Spherical vector components -> Cartesian components.
    """

    e_r, e_theta, e_phi = spherical_unit_vectors(theta, phi)

    V = Vr[..., None] * e_r + Vtheta[..., None] * e_theta + Vphi[..., None] * e_phi

    return V[..., 0], V[..., 1], V[..., 2]


def sph_to_cart_vf(A, th, phi):
    """Wrapper for spherical_vector_to_cartesian."""

    return spherical_vector_to_cartesian(A[..., 0], A[..., 1], A[..., 2], th, phi)


# OLD DEFINITION
# def sph_to_cart_vf(A, th, phi):
#     """Rotate spherical vector-field components into Cartesian components."""

#     if isinstance(th, list) and isinstance(phi, list):
#         points_rot = [cart_to_sph_rotation_mat(theta, phii).T for theta, phii in zip(th, phi)]
#         rotated_to_cart = [np.dot(rot_m, point)
#               for rot_m, point in zip(points_rot, A)]
#         return np.array(rotated_to_cart)
#     else:
#         rot_m = cart_to_sph_rotation_mat(th, phi).T
#         rotated_to_cart = np.dot(rot_m, A)
#         return np.asarray(rotated_to_cart)


def Rot(th, axis="x"):
    """Return a 3D rotation matrix about the requested axis.

    Parameters
    ----------
    th : float
        Rotation angle in radians.
    axis : {'x', 'y', 'z'}, default='x'
        Axis about which to rotate.
    """

    if axis == "x":
        rot_m = [[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]]
    elif axis == "y":
        rot_m = [[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]
    elif axis == "z":
        rot_m = [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
    else:
        raise ValueError("axis must be one of 'x', 'y', or 'z'")

    return np.array(rot_m)


# OLD DEFINITION
# def Rot(th, axis="x"):
#     """Return a 3D rotation matrix about the requested axis.

#     Parameters
#     ----------
#     th : float
#         Rotation angle in radians.
#     axis : {'x', 'y', 'z'}, default='x'
#         Axis about which to rotate.
#     """

#     if axis == "x":
#         rot_m = [[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]]
#     elif axis == "y":
#         rot_m = [[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]
#     elif axis == "z":
#         rot_m = [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
#     else:
#         raise ValueError("axis must be one of 'x', 'y', or 'z'")

#     return np.array(rot_m)


# Mapping of old functions on to new:
# cart_to_sph -> spherical_unit_vectors
# cart_to_sph_vf -> cartesian_vector_to_spherical
# sph_to_cart_vf -> spherical_vector_to_cartesian
