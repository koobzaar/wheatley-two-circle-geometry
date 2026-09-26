"""A6: area of the two-circle family with height H and vertical profile g (C-024, C-025)."""
import numpy as np
import pytest

from wheatley import nleaflet as nl


def surf1(t, s, H, g):
    b = g(s) / 2
    return np.array([-b + (1 - b) * np.cos(t), (1 - b) * np.sin(t), H * s])


def surf2(t, s, H, g, th):
    b = g(s) / 2
    x, y = -(1 - b) + b * np.cos(t), b * np.sin(t)
    return np.array([np.cos(th) * x - np.sin(th) * y, np.sin(th) * x + np.cos(th) * y, H * s])


def fd_density(X, t, s, h=1e-6):
    Xt = (X(t + h, s) - X(t - h, s)) / (2 * h)
    Xs = (X(t, s + h) - X(t, s - h)) / (2 * h)
    return np.linalg.norm(np.cross(Xt, Xs))


@pytest.mark.parametrize("H", [0.5, 1.0, 1.5, 3.0])
@pytest.mark.parametrize("m", [0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("N", [2, 3, 4, 6])
def test_density_formula(H, m, N):                                      # C-024
    g = lambda s: s ** m
    dg = lambda s: m * s ** (m - 1)
    th = 2 * np.pi / N
    rng = np.random.default_rng(0)
    for t, s in zip(rng.uniform(0, 2 * np.pi, 20), rng.uniform(0.05, 0.95, 20)):
        lhs = fd_density(lambda t, s: surf1(t, s, H, g), t, s) + fd_density(lambda t, s: surf2(t, s, H, g, th), t, s)
        rhs = 0.5 * np.sqrt(4 * H * H + dg(s) ** 2 * (1 + np.cos(t)) ** 2)
        assert np.isclose(lhs, rhs, rtol=1e-6, atol=1e-8)


def test_two_leaflets_junction_parameter_is_constant():                  # used by C-025
    for z in np.linspace(0.0, 1.0, 11):
        assert np.isclose(float(nl.t_M(2, z)), 2 * np.pi)


@pytest.mark.parametrize("H", [0.5, 1.0, 1.5])
def test_two_leaflets_linear_profile_minimises_area(H):                 # C-025 (numeric check of the Jensen bound)
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from a6_area_profile import area
    lin = area(2, H, 1.0)
    for m in (0.5, 0.8, 1.25, 2.0, 5.0):
        assert area(2, H, m) > lin
    assert 2 * area(2, H, 40.0) < np.pi * (2 * H + 1)                  # supremum (wall limit) not exceeded


@pytest.mark.parametrize("N", [2, 3, 4, 5, 6, 7, 12])
def test_swept_area_is_pi_over_N(N):                                     # C-027: W_N = theta/2 (Green)
    from scipy.integrate import quad
    W = 0.5 * quad(lambda z: (float(nl.t_M(N, z)) - np.pi) + np.sin(float(nl.t_M(N, z))), 0, 1,
                   epsabs=1e-13, epsrel=1e-13, limit=200)[0]
    assert np.isclose(N * W, np.pi, rtol=0, atol=1e-12)


def test_projected_jacobians_sum():                                      # C-027: |J1| + |J2| = (1 + cos t)/2
    rng = np.random.default_rng(1)
    for t, z in zip(rng.uniform(0, 2 * np.pi, 50), rng.uniform(0, 1, 50)):
        b = z / 2
        a = 1 - b
        J1 = (-a * np.sin(t)) * (-0.5 * np.sin(t)) - (a * np.cos(t)) * (-0.5 - 0.5 * np.cos(t))
        J2 = (-b * np.sin(t)) * (0.5 * np.sin(t)) - (b * np.cos(t)) * (0.5 + 0.5 * np.cos(t))
        assert J1 >= -1e-15 and J2 <= 1e-15
        assert np.isclose(abs(J1) + abs(J2), 0.5 * (1 + np.cos(t)))
