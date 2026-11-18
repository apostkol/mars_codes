from __future__ import annotations

import numpy as np
import pytest

from mars_currents.generic import (
    Rot,
    bow_shock,
    cartesian_to_spherical,
    cartesian_vector_to_spherical,
    spherical_to_cartesian,
    spherical_unit_vectors,
    spherical_vector_to_cartesian,
)

# ============================================================
# Helpers
# ============================================================

RTOL = 1e-7
ATOL = 1e-7


def assert_allclose(a, b):
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)


def test_bow_shock_accepts_array_angles():
    angles = np.array([0.0, np.pi / 2])

    radii = bow_shock(x0=0.74, L=1.82, eps=1.01, th=angles)

    assert np.allclose(radii, [1.82 / 2.01, 1.82])


def test_rotation_matrix_z_axis():
    rotation = Rot(np.pi / 2, axis="z")
    vector = np.array([1.0, 0.0, 0.0])

    rotated = rotation @ vector

    assert np.allclose(rotated, [0.0, 1.0, 0.0])


# ============================================================
# Coordinate transform tests
# ============================================================


@pytest.mark.parametrize(
    "xyz, expected",
    [
        ((1, 0, 0), (1, np.pi / 2, 0)),
        ((0, 1, 0), (1, np.pi / 2, np.pi / 2)),
        ((0, 0, 1), (1, 0, 0)),
        ((0, 0, -1), (1, np.pi, 0)),
    ],
)
def test_cartesian_to_spherical_known_values(xyz, expected):
    x, y, z = xyz
    r_exp, theta_exp, phi_exp = expected

    r, theta, phi = cartesian_to_spherical(x, y, z)

    assert_allclose(r, r_exp)
    assert_allclose(theta, theta_exp)
    assert_allclose(phi, phi_exp)


@pytest.mark.parametrize(
    "rtp",
    [
        (1, np.pi / 2, 0),
        (1, np.pi / 2, np.pi / 2),
        (1, 0, 0),
        (2, np.pi / 3, np.pi / 4),
    ],
)
def test_spherical_to_cartesian_roundtrip(rtp):
    r, theta, phi = rtp

    x, y, z = spherical_to_cartesian(r, theta, phi)
    r2, theta2, phi2 = cartesian_to_spherical(x, y, z)

    assert_allclose(r2, r)
    assert_allclose(theta2, theta)

    # periodic angle handling
    assert_allclose(np.mod(phi2, 2 * np.pi), np.mod(phi, 2 * np.pi))


def test_cartesian_spherical_vectorized_roundtrip():
    rng = np.random.default_rng(1234)

    x = rng.normal(size=(100,))
    y = rng.normal(size=(100,))
    z = rng.normal(size=(100,))

    r, theta, phi = cartesian_to_spherical(x, y, z)

    x2, y2, z2 = spherical_to_cartesian(r, theta, phi)

    assert_allclose(x2, x)
    assert_allclose(y2, y)
    assert_allclose(z2, z)


# ============================================================
# Basis vector tests
# ============================================================


def test_basis_vectors_are_orthonormal():
    theta = 1.1
    phi = 0.7

    e_r, e_theta, e_phi = spherical_unit_vectors(theta, phi)

    # unit length
    assert_allclose(np.linalg.norm(e_r), 1.0)
    assert_allclose(np.linalg.norm(e_theta), 1.0)
    assert_allclose(np.linalg.norm(e_phi), 1.0)

    # orthogonality
    assert_allclose(np.dot(e_r, e_theta), 0.0)
    assert_allclose(np.dot(e_r, e_phi), 0.0)
    assert_allclose(np.dot(e_theta, e_phi), 0.0)


def test_basis_vectors_right_handed():
    theta = 0.9
    phi = 1.2

    e_r, e_theta, e_phi = spherical_unit_vectors(theta, phi)

    cross = np.cross(e_r, e_theta)

    assert_allclose(cross, e_phi)


# ============================================================
# Vector transform tests
# ============================================================


def test_cartesian_vector_to_spherical_known_case():
    """
    At theta=pi/2, phi=0:

        e_r     = x_hat
        e_theta = -z_hat
        e_phi   = y_hat
    """

    theta = np.pi / 2
    phi = 0.0

    Vx, Vy, Vz = 1.0, 2.0, 3.0

    Vr, Vtheta, Vphi = cartesian_vector_to_spherical(Vx, Vy, Vz, theta, phi)

    assert_allclose(Vr, 1.0)
    assert_allclose(Vtheta, -3.0)
    assert_allclose(Vphi, 2.0)


def test_vector_component_roundtrip_scalar():
    rng = np.random.default_rng(0)

    Vx, Vy, Vz = rng.normal(size=3)

    theta = 1.1
    phi = 0.8

    Vr, Vtheta, Vphi = cartesian_vector_to_spherical(Vx, Vy, Vz, theta, phi)

    Vx2, Vy2, Vz2 = spherical_vector_to_cartesian(Vr, Vtheta, Vphi, theta, phi)

    assert_allclose(Vx2, Vx)
    assert_allclose(Vy2, Vy)
    assert_allclose(Vz2, Vz)


def test_vector_component_roundtrip_vectorized():
    rng = np.random.default_rng(42)

    shape = (32, 16)

    Vx = rng.normal(size=shape)
    Vy = rng.normal(size=shape)
    Vz = rng.normal(size=shape)

    theta = rng.uniform(0.1, np.pi - 0.1, size=shape)
    phi = rng.uniform(0, 2 * np.pi, size=shape)

    Vr, Vtheta, Vphi = cartesian_vector_to_spherical(Vx, Vy, Vz, theta, phi)

    Vx2, Vy2, Vz2 = spherical_vector_to_cartesian(Vr, Vtheta, Vphi, theta, phi)

    assert_allclose(Vx2, Vx)
    assert_allclose(Vy2, Vy)
    assert_allclose(Vz2, Vz)


# ============================================================
# Shape / broadcasting tests
# ============================================================


def test_vectorized_shapes_preserved():
    x = np.zeros((8, 4))
    y = np.ones((8, 4))
    z = np.ones((8, 4))

    r, theta, phi = cartesian_to_spherical(x, y, z)

    assert r.shape == (8, 4)
    assert theta.shape == (8, 4)
    assert phi.shape == (8, 4)


def test_basis_vector_shapes():
    theta = np.ones((5, 7))
    phi = np.ones((5, 7))

    e_r, e_theta, e_phi = spherical_unit_vectors(theta, phi)

    assert e_r.shape == (5, 7, 3)
    assert e_theta.shape == (5, 7, 3)
    assert e_phi.shape == (5, 7, 3)


# ============================================================
# Edge cases
# ============================================================


def test_origin_behavior():
    r, theta, phi = cartesian_to_spherical(0.0, 0.0, 0.0)

    assert_allclose(r, 0.0)

    # theta/phi formally undefined at origin;
    # test only finite output
    assert np.isfinite(theta)
    assert np.isfinite(phi)


def test_phi_range():
    x = np.array([-1, -1, 1, 1])
    y = np.array([1, -1, -1, 1])
    z = np.zeros(4)

    _, _, phi = cartesian_to_spherical(x, y, z)

    assert np.all(phi >= 0)
    assert np.all(phi < 2 * np.pi)
