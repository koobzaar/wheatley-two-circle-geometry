"""Numerical twins of lean/WheatleyGeom/NLeaflet.lean (C-003, C-012, C-013, C-014, C-015)."""

import numpy as np
import pytest

from wheatley import geometry as g
from wheatley import nleaflet as nl

NS = [2, 3, 4, 5, 6, 7]
ZS = np.linspace(0.0, 1.0, 21)


def _fd_normal(f, t, z, h=1e-6):
    Xt = (f(t + h, z) - f(t - h, z)) / (2 * h)
    Xz = (f(t, z + h) - f(t, z - h)) / (2 * h)
    return np.cross(Xt, Xz)


@pytest.mark.parametrize("N", NS)
def test_same_parameter_junction(N):                                    # C-012
    for z in ZS:
        t = nl.t_M(N, z)
        assert np.allclose(g.surface1(t, z), nl.surface2(N, t, z), atol=1e-12)
        assert np.pi - 1e-12 <= t <= 2 * np.pi + 1e-12


def test_three_matches_paper():                                         # tMθ_two_pi_div_three
    assert np.allclose(nl.t_M(3, ZS), g.t_M(ZS), atol=1e-13)
    assert np.allclose(nl.surface2(3, 4.2, ZS), g.surface2(4.2, ZS), atol=1e-13)


@pytest.mark.parametrize("N", NS)
def test_junction_endpoints(N):                                         # M(0) = R_θ E, M(1) = O
    assert np.isclose(nl.t_M(N, 0.0), np.pi + 2 * np.pi / N)
    assert np.isclose(nl.t_M(N, 1.0), 2 * np.pi)
    assert np.allclose(g.surface1(nl.t_M(N, 1.0), 1.0), [0, 0, 1], atol=1e-12)


@pytest.mark.parametrize("N", NS)
def test_top_edge_closes(N):                                            # C-013
    for t in np.linspace(np.pi, 2 * np.pi, 11):
        assert np.allclose(nl.surface2(N, t, 1.0), g.rot_ccw(g.surface1(t, 1.0), 2 * np.pi / N))


@pytest.mark.parametrize("N", NS)
def test_fold_cos_formula_matches_normals(N):                           # C-014
    for z in np.linspace(0.05, 1.0, 12):
        t = nl.t_M(N, z)
        n1 = _fd_normal(g.surface1, t, z)
        n2 = _fd_normal(lambda a, b: nl.surface2(N, a, b), t, z)
        c = (n1 @ -n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
        assert np.isclose(c, nl.fold_cos(N, z), atol=1e-7)


def test_fold_three_closed_form_and_bounds():                          # C-003
    z = np.linspace(1e-6, 1.0, 101)
    D = 1 - z / 2 + (z / 2) ** 2
    assert np.allclose(nl.fold_cos(3, z), (8 * D**2 + 9) / (16 * D**2 + 9), atol=1e-14)
    ang = nl.fold_angle_deg(3, z)
    assert ang.min() > 41.40 and ang.max() < 47.16
    assert np.isclose(nl.fold_angle_deg(3, 1.0), np.degrees(np.arccos(0.75)))


def test_two_leaflets_tangent_continuous():                            # C-014, N = 2
    assert np.allclose(nl.fold_cos(2, np.linspace(0.01, 1, 50)), 1.0)
    assert np.allclose(nl.t_M(2, np.linspace(0, 1, 11)), 2 * np.pi)


@pytest.mark.parametrize("N", NS)
def test_top_fold_angle(N):                                             # cos δ(z=1) = sin²(π/N)
    assert np.isclose(nl.fold_cos(N, 1.0), np.sin(np.pi / N) ** 2)


def test_area_not_preserved():                                          # C-015
    # numerical evidence only, for the tested values N = 2..30 (no proof for every N)
    A = {N: N * nl.leaflet_area(N) for N in range(2, 31)}
    assert np.isclose(nl.leaflet_area(3), 2.89364, atol=1e-4)
    assert all(A[N + 1] > A[N] for N in range(2, 30))


def test_fold_range_four_leaflets():                                    # C-014, N = 4
    # delta in [60 deg, arccos(1/5)): minimum at z = 1, supremum approached (not attained) as z -> 0
    sup = np.degrees(np.arccos(0.2))
    assert np.isclose(sup, 78.463040967, atol=1e-8)
    assert np.isclose(nl.fold_angle_deg(4, 1.0), 60.0)
    zs = np.concatenate([[1e-6, 1e-3], np.linspace(0.01, 1.0, 200)])
    d = nl.fold_angle_deg(4, zs)
    assert np.all(d < sup) and np.all(d >= 60.0 - 1e-9)
    assert sup - nl.fold_angle_deg(4, 1e-6) < 1e-3
