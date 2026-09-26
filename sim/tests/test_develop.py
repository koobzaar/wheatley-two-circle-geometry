"""Numerical cross-checks of front A1 (flat patterns), independent of the Lean proofs.

C-009 (congruence), C-010 (flat pattern is an isometry, no overlap), C-011 (junction images have
the same length but different geodesic curvature: no single-sheet curved fold).
"""

import numpy as np
import pytest

from wheatley import develop as d
from wheatley import geometry as g

SQ3 = np.sqrt(3.0)


def _fd_first_form(F, t, z, h=1e-6):
    Ft = (F(t + h, z) - F(t - h, z)) / (2 * h)
    Fz = (F(t, z + h) - F(t, z - h)) / (2 * h)
    return np.array([Ft @ Ft, Ft @ Fz, Fz @ Fz])


def _embed(surface):
    def F(t, z):
        p = d.develop(surface, np.array([t]), np.array([z]))[0]
        return np.array([p[0], p[1], 0.0])
    return F


@pytest.mark.parametrize("surface", [1, 2])
@pytest.mark.parametrize("t,z", [(3.3, 0.2), (4.0, 0.5), (5.0, 0.9), (5.9, 0.95), (4.5, 1.5)])
def test_development_is_isometry(surface, t, z):                       # C-010 (numeric twin)
    X = g.surface1 if surface == 1 else g.surface2
    I_surf = _fd_first_form(lambda a, b: X(a, b), t, z)
    I_dev = _fd_first_form(_embed(surface), t, z)
    assert np.allclose(I_surf, I_dev, rtol=1e-6, atol=1e-7)


def test_dev_speed_matches_definition():                               # |w' x w| / |w|^2
    for t in np.linspace(np.pi, 2 * np.pi, 13):
        w1 = np.array([np.cos(t) + 1, np.sin(t), -2.0])
        w1p = np.array([-np.sin(t), np.cos(t), 0.0])
        v = np.linalg.norm(np.cross(w1p, w1)) / (w1 @ w1)
        assert np.isclose(v, d.dev_speed(t), rtol=1e-13)
        assert d.dev_speed(t) <= 0.5 + 1e-15                           # devSpeed1_le


def test_total_angle_below_half_pi():                                  # devAngle1_mem
    zz = np.linspace(0, 1, 21)
    ph = d.phi(g.t_M(zz))
    assert np.all(ph >= 0) and np.all(ph <= np.pi / 2)
    assert np.all(np.diff(ph) > 0)


@pytest.mark.parametrize("surface", [1, 2])
def test_flat_pattern_no_overlap(surface):                             # develop_injOn (numeric)
    G = d.develop_grid(surface, nt=40, nz=30, z_min=0.02)
    P = G.reshape(-1, 2)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    assert D.min() > 1e-4


def test_congruence():                                                 # C-009
    def T(p):
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        return np.stack([-x / 2 - SQ3 / 2 * y, SQ3 / 2 * x - y / 2, 2 - z], axis=-1)
    for t in np.linspace(0, 2 * np.pi, 9):
        for z in np.linspace(-1, 3, 9):
            assert np.allclose(T(g.surface1(t, z)), g.surface2(t, 2 - z), atol=1e-13)


def test_junction_images_same_length():                                # C-010 consequence
    z = np.linspace(0, 1, 4001)
    L3 = d.arclength(g.junction_point(z))[-1]
    for s in (1, 2):
        assert np.isclose(d.arclength(d.junction_image(s, z))[-1], L3, rtol=1e-6)


def _kg(z, surf, h=1e-5):
    f = g.surface1 if surf == 1 else g.surface2
    tm = g.t_M(z)
    G1 = (g.junction_point(z + h) - g.junction_point(z - h)) / (2 * h)
    G2 = (g.junction_point(z + h) - 2 * g.junction_point(z) + g.junction_point(z - h)) / h**2
    sp = np.linalg.norm(G1)
    T = G1 / sp
    acc = (G2 - (G2 @ T) * T) / sp**2
    nu = -(f(tm + h, z) - f(tm - h, z)) / (2 * h)
    nu = nu - (nu @ T) * T
    nu /= np.linalg.norm(nu)
    return acc @ nu


@pytest.mark.parametrize("z", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_junction_geodesic_curvatures_do_not_cancel(z):               # C-011 (numeric)
    s = _kg(z, 1) + _kg(z, 2)
    assert s < -0.01
