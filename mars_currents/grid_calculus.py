"""Finite-difference helpers for spherical grid fields."""

from __future__ import annotations

import warnings

import numpy as np

np.seterr(divide="ignore")


def _resolved_neighbour(center, coordinate, direction, field):
    neighbour = center.neighbours[coordinate][direction]

    if neighbour is None:
        return center

    if np.isnan(getattr(neighbour, field + "r")):
        return center

    return neighbour


def _signed_angular_separation(plus, minus):
    """Return the shortest signed separation from ``minus`` to ``plus``.

    Unlike direct subtraction, this remains a local positive stencil width
    when the neighbours straddle the periodic -pi/pi (or 0/2pi) seam.
    """
    difference = plus - minus
    return np.arctan2(np.sin(difference), np.cos(difference))


def curl(center, field="B"):
    """Estimate the curl of a vector field at a spherical grid cell.

    Parameters
    ----------
    center : Cell
        Central cell. Neighbour values are read from ``center.neighbours``.
    field : str
        Prefix for the vector component names, such as ``"B"`` or ``"J"``.

    Returns
    -------
    list[float]
        The ``r``, ``theta``, and ``phi`` components of the curl.
    """

    r_minus = _resolved_neighbour(center, "r", -1, field)
    r_plus = _resolved_neighbour(center, "r", +1, field)
    th_minus = _resolved_neighbour(center, "th", -1, field)
    th_plus = _resolved_neighbour(center, "th", +1, field)
    phi_minus = _resolved_neighbour(center, "phi", -1, field)
    phi_plus = _resolved_neighbour(center, "phi", +1, field)

    th0 = center.th_c
    r0 = center.r_c
    dphi = _signed_angular_separation(phi_plus.phi_c, phi_minus.phi_c)

    curl_r = (
        np.cos(th0) / (r0 * np.sin(th0)) * getattr(center, field + "phi")
        + 1
        / r0
        * (getattr(th_plus, field + "phi") - getattr(th_minus, field + "phi"))
        / (th_plus.th_c - th_minus.th_c)
        - 1
        / (r0 * np.sin(th0))
        * (getattr(phi_plus, field + "th") - getattr(phi_minus, field + "th"))
        / dphi
    )

    curl_th = (
        1
        / (r0 * np.sin(th0))
        * (getattr(phi_plus, field + "r") - getattr(phi_minus, field + "r"))
        / dphi
        - getattr(center, field + "phi") / r0
        - (getattr(r_plus, field + "phi") - getattr(r_minus, field + "phi"))
        / (r_plus.r_c - r_minus.r_c)
    )

    curl_phi = (
        getattr(center, field + "th") / r0
        + (getattr(r_plus, field + "th") - getattr(r_minus, field + "th"))
        / (r_plus.r_c - r_minus.r_c)
        - 1
        / r0
        * (getattr(th_plus, field + "r") - getattr(th_minus, field + "r"))
        / (th_plus.th_c - th_minus.th_c)
    )

    return [curl_r, curl_th, curl_phi]


def div(center, field="B"):
    """Estimate the divergence of a vector field at a spherical grid cell.

    Missing neighbours are replaced by the central cell, matching the behavior
    used at grid boundaries.
    """

    r_minus = _resolved_neighbour(center, "r", -1, field)
    r_plus = _resolved_neighbour(center, "r", +1, field)
    th_minus = _resolved_neighbour(center, "th", -1, field)
    th_plus = _resolved_neighbour(center, "th", +1, field)
    phi_minus = _resolved_neighbour(center, "phi", -1, field)
    phi_plus = _resolved_neighbour(center, "phi", +1, field)

    th0 = center.th_c
    r0 = center.r_c
    dphi = _signed_angular_separation(phi_plus.phi_c, phi_minus.phi_c)

    divergence = (
        1
        / r0
        * (
            2 * getattr(center, field + "r")
            + (getattr(th_plus, field + "th") - getattr(th_minus, field + "th"))
            / (th_plus.th_c - th_minus.th_c)
        )
        + 1
        / (r0 * np.sin(th0))
        * (
            np.cos(th0) * getattr(center, field + "th")
            + (getattr(phi_plus, field + "phi") - getattr(phi_minus, field + "phi"))
            / dphi
        )
        + (getattr(r_plus, field + "r") - getattr(r_minus, field + "r"))
        / (r_plus.r_c - r_minus.r_c)
    )

    return divergence


def sigma_gradient(center, field, comp, resp):
    """Estimate uncertainty in a finite-difference gradient.

    Parameters use the package's component naming convention. For example,
    ``field="B"`` and ``comp="r"`` reads ``sigma_Br`` from the center and its
    neighbours. ``resp`` selects the coordinate direction with respect to which
    the gradient is taken.
    """

    minus = _resolved_neighbour(center, resp, -1, field)
    plus = _resolved_neighbour(center, resp, +1, field)
    sigma_attr = "sigma_" + field + comp

    sigma_gr = np.sqrt(
        getattr(minus, sigma_attr) ** 2
        + 2 * getattr(center, sigma_attr) ** 2
        + getattr(plus, sigma_attr) ** 2,
    ) / (np.sqrt(2) * getattr(center, "d" + resp))

    return sigma_gr


def radial_profile(
    grid,
    value="Jr",
    reducer=np.nanmean,
    latitude_deg=None,
    longitude_deg=None,
):
    """Extract a radial profile from a 3D grid quantity.

    Grid-level quantities such as ``Jr`` are stored with array order
    ``(phi, theta, r)``. This helper returns ``(grid.r, profile)`` after either
    reducing over all angular bins or selecting the nearest latitude and/or
    longitude before reducing.

    Parameters
    ----------
    grid : Grid
        Grid with a 3D array attribute such as ``Jr`` or ``Br``.
    value : str, default='Jr'
        Name of the grid attribute to profile.
    reducer : callable, default=numpy.nanmean
        Function used to reduce angular bins. It must accept an ``axis``
        keyword. Use ``np.nanmedian`` or a custom callable when appropriate.
    latitude_deg : float, optional
        Planetographic-style latitude in degrees. Internally this is converted
        to grid colatitude ``theta = 90 - latitude``.
    longitude_deg : float, optional
        Longitude in degrees. Values are wrapped to ``[0, 360)`` before matching
        the nearest grid azimuth.
    """

    if not hasattr(grid, value):
        raise AttributeError(f"Grid has no array named {value!r}")

    data = np.asarray(getattr(grid, value), dtype=float)
    expected_shape = (len(grid.phi), len(grid.th), len(grid.r))
    if data.shape != expected_shape:
        raise ValueError(
            f"{value!r} has shape {data.shape}; expected {expected_shape} "
            "for grid array order (phi, theta, r).",
        )

    phi_selection = slice(None)
    if longitude_deg is not None:
        longitude_rad = np.deg2rad(longitude_deg % 360)
        phi_delta = (grid.phi - longitude_rad + np.pi) % (2 * np.pi) - np.pi
        phi_index = int(np.nanargmin(np.abs(phi_delta)))
        phi_selection = slice(phi_index, phi_index + 1)

    th_selection = slice(None)
    if latitude_deg is not None:
        theta_rad = np.deg2rad(90 - latitude_deg)
        th_index = int(np.nanargmin(np.abs(grid.th - theta_rad)))
        th_selection = slice(th_index, th_index + 1)

    selected = data[phi_selection, th_selection, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        profile = reducer(selected, axis=(0, 1))

    return np.asarray(grid.r), np.asarray(profile)
