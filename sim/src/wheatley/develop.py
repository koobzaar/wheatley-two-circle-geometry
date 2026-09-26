"""Flat patterns (planar developments) of the two cone patches of the leaflet -- front A1.

Mirrors lean/WheatleyGeom/Cone.lean + Leaflet.lean (claims C-009, C-010):
  cone map      X(t, z) = V + (alpha + beta z) w(t)
  development   D(t, z) = (alpha + beta z) |w(t)| (cos phi(t), sin phi(t)),
                phi(t)  = int_pi^t |w' x w| / |w|^2 ds.
For both surfaces |w|^2 = 6 + 2 cos t and |w' x w|^2 = 4 + (1 + cos t)^2 (Leaflet.lean:
dot_w1, dot_w2, dot_cross_w1, dot_cross_w2), so phi is the SAME function for both.
Surface 1: alpha = 1, beta = -1/2 (radius (1 - z/2) r(t)); surface 2: alpha = 0, beta = 1/2.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from . import geometry as g


def r(t):
    """|w_i(t)| = sqrt(6 + 2 cos t), i = 1, 2 (generator length from the apex to the directrix)."""
    return np.sqrt(6.0 + 2.0 * np.cos(t))


def dev_speed(t):
    """phi'(t) = |w' x w| / |w|^2 = sqrt(4 + (1 + cos t)^2) / (6 + 2 cos t)."""
    c = np.cos(t)
    return np.sqrt(4.0 + (1.0 + c) ** 2) / (6.0 + 2.0 * c)


def phi(t):
    """Development angle phi(t) = int_pi^t dev_speed (same for surfaces 1 and 2)."""
    t = np.atleast_1d(np.asarray(t, float))
    out = np.array([quad(dev_speed, np.pi, tt, epsabs=1e-13, epsrel=1e-13)[0] for tt in t])
    return out


def develop(surface: int, t, z):
    """Planar image (x, y) of X_surface(t, z)."""
    t, z = np.broadcast_arrays(np.asarray(t, float), np.asarray(z, float))
    f = (1.0 - z / 2.0) if surface == 1 else (z / 2.0)
    rho = f * r(t)
    ph = phi(t.ravel()).reshape(t.shape)
    return np.stack([rho * np.cos(ph), rho * np.sin(ph)], axis=-1)


def develop_grid(surface: int, nt: int = 80, nz: int = 60, z_min: float = 0.0):
    """Planar images of the Omega grid of one surface (for plots and area checks)."""
    z = np.linspace(z_min, 1.0, nz)
    out = np.empty((nz, nt, 2))
    for i, zz in enumerate(z):
        tt = np.linspace(np.pi, g.t_M(zz), nt)
        out[i] = develop(surface, tt, np.full_like(tt, zz))
    return out


def junction_image(surface: int, z):
    """Image of the junction curve M(z) (t = t_M(z)) in the flat pattern of `surface`."""
    z = np.asarray(z, float)
    return develop(surface, g.t_M(z), z)


def planar_curvature(P, s):
    """Signed curvature of a planar curve P(s) sampled at parameters s (finite differences)."""
    d1 = np.gradient(P, s, axis=0)
    d2 = np.gradient(d1, s, axis=0)
    num = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    return num / np.linalg.norm(d1, axis=1) ** 3


def arclength(P):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])
