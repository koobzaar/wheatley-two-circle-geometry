"""Fold angle under a vertical stretch (x, y, z) -> (x, y, h z)  (CONVENTIONS A-13, claim C-021).

Closed form: cos delta_h = (X^2 - 4 h^2 cos th) / (X^2 + 4 h^2), X = 1 + cos t_M = (1 - cos th)/Delta.
Checked against finite-difference normals of the stretched surfaces."""

import numpy as np
import pytest

from wheatley import geometry as g
from wheatley import nleaflet as nl


def _n(f, t, z, h, e=1e-6):
    S = lambda a, b: f(a, b) * np.array([1.0, 1.0, h])
    return np.cross((S(t + e, z) - S(t - e, z)) / (2 * e), (S(t, z + e) - S(t, z - e)) / (2 * e))


@pytest.mark.parametrize("N", [2, 3, 4, 6])
@pytest.mark.parametrize("h", [0.5, 1.0, 1.5, 2.0])
def test_fold_cos_with_vertical_scale(N, h):
    th = 2 * np.pi / N
    for z in np.linspace(0.05, 1.0, 9):
        t = float(nl.t_M(N, z))
        n1 = _n(g.surface1, t, z, h)
        n2 = _n(lambda a, b: nl.surface2(N, a, b), t, z, h)
        c = (n1 @ -n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
        X = (1 - np.cos(th)) / nl.delta(N, z)
        assert np.isclose(c, (X**2 - 4 * h**2 * np.cos(th)) / (X**2 + 4 * h**2), atol=1e-7)


def test_top_value_matches_A13():
    for h in (1.0, 1.5):
        c = (4 + 2 * h**2) / (4 + 4 * h**2)
        assert np.isclose(c, (h**2 + 2) / (2 * (h**2 + 1)))
    assert np.isclose(np.degrees(np.arccos((1.5**2 + 2) / (2 * (1.5**2 + 1)))), 49.1677829725, atol=1e-8)
