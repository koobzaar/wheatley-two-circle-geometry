"""N-leaflet generalisation of the two-circle construction (fronts A3, A2, A4).

Mirrors lean/WheatleyGeom/NLeaflet.lean.  theta = 2 pi / N.  Surface 1 is geometry.surface1;
surface 2 is the small circle (15-16), b = z/2, rotated counter-clockwise by theta.
  junction parameter  cos t_M = (2ab - (a^2+b^2) cos th)/Delta,  sin t_M = (b^2-a^2) sin th/Delta,
                      Delta = a^2 + b^2 - 2ab cos th,  a = 1 - b;
  fold angle          cos delta = (X^2 - 4 cos th)/(X^2 + 4),  X = 1 + cos t_M = (1 - cos th)/Delta.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from . import geometry as g


def surface2(N: int, t, z):
    """Small circle (15)-(16) with b = z/2, rotated by 2 pi / N (NLeaflet.lean: surf2θ)."""
    t, z = np.broadcast_arrays(np.asarray(t, float), np.asarray(z, float))
    b = z / 2
    p = np.stack([-(1 - b) + b * np.cos(t), b * np.sin(t), z], axis=-1)
    return g.rot_ccw(p, 2 * np.pi / N)


def delta(N: int, z):
    th = 2 * np.pi / N
    b = np.asarray(z, float) / 2
    a = 1 - b
    return a * a + b * b - 2 * a * b * np.cos(th)


def t_M(N: int, z):
    """Junction parameter tMθ = 2 pi - arccos(cMθ) (NLeaflet.lean)."""
    th = 2 * np.pi / N
    b = np.asarray(z, float) / 2
    a = 1 - b
    c = (2 * a * b - (a * a + b * b) * np.cos(th)) / delta(N, z)
    return 2 * np.pi - np.arccos(np.clip(c, -1.0, 1.0))


def fold_cos(N: int, z):
    """cos of the fold angle along the junction (foldCos_junction); 1 = tangent-continuous."""
    th = 2 * np.pi / N
    X = (1 - np.cos(th)) / delta(N, z)
    return (X * X - 4 * np.cos(th)) / (X * X + 4)


def fold_angle_deg(N: int, z):
    return np.degrees(np.arccos(np.clip(fold_cos(N, z), -1.0, 1.0)))


def leaflet_area(N: int) -> float:
    """Area of one leaflet (surface 1 + surface 2 over Omega_N), height scale h = 1.

    |X1_t x X1_z| + |X2_t x X2_z| = (1/2) sqrt(4 + (1 + cos t)^2)  (independent of z)."""
    f = lambda t: np.sqrt(4.0 + (1.0 + np.cos(t)) ** 2)
    return 0.5 * quad(lambda z: quad(f, np.pi, float(t_M(N, z)), epsabs=1e-12)[0], 0.0, 1.0,
                      epsabs=1e-11)[0]
