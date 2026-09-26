"""Numerical checks backing the entries in CLAIMS.md."""

import numpy as np
import pytest

from wheatley import geometry as g

Z = np.linspace(0.05, 0.95, 19)


@pytest.mark.parametrize("z", Z)
def test_junction_point_lies_on_both_surfaces(z):
    M = g.junction_point(z)
    assert np.allclose(g.surface1(g.t_range_surface1(z)[1], z), M, atol=1e-12)
    assert np.allclose(g.surface2(g.t_range_surface2(z)[0], z), M, atol=1e-12)


@pytest.mark.parametrize("z", Z)
def test_arc_endpoints(z):
    assert np.allclose(g.surface1(np.pi, z)[:2], [-1, 0])                    # E
    assert np.allclose(g.surface2(np.pi, z)[:2], [0.5, -np.sqrt(3) / 2])     # P


def test_surface1_is_cone_with_apex():                                         # C-001
    for t in np.linspace(np.pi, 1.9 * np.pi, 7):
        L = g.ruling(1, t)
        d = L[-1] - L[0]
        s = (g.APEX_SURFACE1 - L[0]) @ d / (d @ d)
        assert np.allclose(L[0] + s * d, g.APEX_SURFACE1, atol=1e-12)


def test_surface2_is_cone_with_apex():                                         # C-001, C-004
    for t in np.linspace(1.1 * np.pi, 1.9 * np.pi, 7):
        L = g.ruling(2, t)
        d = L[-1] - L[0]
        s = (g.APEX_SURFACE2 - L[0]) @ d / (d @ d)
        assert np.allclose(L[0] + s * d, g.APEX_SURFACE2, atol=1e-12)


@pytest.mark.parametrize("surface,t", [(1, 1.3 * np.pi), (1, 1.7 * np.pi), (2, 1.4 * np.pi), (2, 1.8 * np.pi)])
@pytest.mark.parametrize("z", [0.2, 0.5, 0.8])
def test_developable(surface, t, z):                                          # C-002
    assert abs(g.gaussian_curvature(surface, t, z)) < 1e-4


def test_crease_is_not_tangent_continuous():                                  # C-003
    ang = g.crease_angle_deg(Z)
    assert np.all(ang > 30), ang
