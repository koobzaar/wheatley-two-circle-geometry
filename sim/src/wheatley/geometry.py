"""Two-circle construction of the Wheatley valve leaflet (McKee et al. 2026).

Every definition cites the equation it implements; the transcription, page numbers and the
ambiguities A-1..A-n live in notes/technote/main.tex (Section 1) (the single source for conventions).

Conventions (CONVENTIONS.md section 2):
  - base ring radius 1; height z in [0, 1]; b = z/2, a = 1 - b (eq. 29; Prop. 1 proof, p. 9;
    Appendix A uses 2*b as the third coordinate);
  - rotations are counter-clockwise, R_theta(x, y) = (x cos - y sin, x sin + y cos) (A-1);
  - leaflet 0 = surface 1 (arc EIM of the large circle, eqs. 38-39)
               + surface 2 (arc MNP of the small circle rotated by 2pi/3, eqs. 40-41 = 36-37);
    both on the parameter domain  pi <= t <= t_M(z),  0 <= z <= 1;
  - leaflets k = 1, 2 are R_{2 pi k/3} of leaflet 0 (Appendix A, `rotate`).
Singular set: surface 2 collapses to its apex P = (1/2, -sqrt3/2, 0) at z = 0 (X_t = 0 there);
surface 1 is regular for every z < 2, hence on the whole domain.
"""

from __future__ import annotations

import numpy as np

SQ3 = np.sqrt(3.0)


def radii(z):
    """(a, b) = (1 - z/2, z/2): eq. (29) with b = z/2 (Prop. 1 proof, p. 9)."""
    b = np.asarray(z, dtype=float) / 2.0
    return 1.0 - b, b


def _D(b):
    """D(b) = 1 - b + b^2, the denominator of eqs. (33), (35) and of t_M."""
    return 1.0 - b + b * b


def rot_ccw(p, angle):
    """R_angle about the z axis, counter-clockwise (CONVENTIONS A-1). p has shape (..., 3)."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.asarray(p, float) @ R.T


rotz = rot_ccw  # backwards-compatible name


# --- junction parameter -----------------------------------------------------------------

def t_M(z):
    """Parameter of M on *both* circles: 2pi - arccos((1+2b-2b^2)/(2D)) = 2pi - arcsin(sqrt3(1-2b)/(2D)).

    End of arc EIM (p. 5, p. 7; Appendix A plot1, arcsin form) and start of arc MNP (p. 6, p. 7;
    Appendix A plot2, arccos form).  The two forms agree for b in [0, 1/2]; t_M runs from 5pi/3
    (z = 0) to 2pi (z = 1).
    """
    _, b = radii(z)
    return 2 * np.pi - np.arccos((1 + 2 * b - 2 * b * b) / (2 * _D(b)))


# --- surface 1: large principal circle, centre (-b, 0), radius 1 - b ------------------------

def surface1(t, z):
    """X1(t, z) = (-b + (1-b) cos t, (1-b) sin t, z): eqs. (38)-(39) = (13)-(14) with a = 1 - b; (43)."""
    t, z = np.broadcast_arrays(np.asarray(t, float), np.asarray(z, float))
    a, b = radii(z)
    return np.stack([-b + a * np.cos(t), a * np.sin(t), z], axis=-1)


def t_range_surface1(z):
    """Arc EIM (p. 7): t from pi (E = (-1, 0)) increasing to t_M(z) (M, eq. 33)."""
    return np.pi, t_M(z)


# --- surface 2: small circle radius b rotated by 2pi/3 (eqs. 36-37 = 40-41 = 19-20) --------

def surface2(t, z):
    """X2(t, z): eqs. (40)-(41) (= 36-37, = 19-20), i.e. R_{2pi/3} of (15)-(16) (A-1)."""
    t, z = np.broadcast_arrays(np.asarray(t, float), np.asarray(z, float))
    _, b = radii(z)
    x = 0.5 * ((1 - b) - b * (np.cos(t) + SQ3 * np.sin(t)))
    y = 0.5 * (-SQ3 * (1 - b) + b * (SQ3 * np.cos(t) - np.sin(t)))
    return np.stack([x, y, z], axis=-1)


def t_range_surface2(z):
    """Arc MNP (p. 7): t from t_M(z) (M) decreasing (clockwise, as in the paper) to pi (P)."""
    return t_M(z), np.pi


def junction_point(z):
    """M(z) = (x_M(b), y_M(b), z), eq. (33): the second intersection of circles (30) and (31)."""
    _, b = radii(z)
    D = _D(b)
    return np.stack([(1 - b - 2 * b * b) / (2 * D), -SQ3 * (1 - b) * (1 - 2 * b) / (2 * D),
                     np.asarray(z, float)], axis=-1)


# --- cone structure ------------------------------------------------------------------------

APEX_SURFACE1 = np.array([-1.0, 0.0, 2.0])             # Prop. 1 / eq. (48)
APEX_SURFACE2 = np.array([0.5, -SQ3 / 2, 0.0])         # = X2(t, 0) = P; R_{2pi/3} of eq. (53)
# The paper prints (1/2, +sqrt3/2, 0) (Prop. 1, p. 10): eqs. (3)-(4) applied as a point map,
# which rotates clockwise. See CONVENTIONS A-1/A-2 and claim C-004.


def ruling(surface: int, t: float, n: int = 20):
    """Segment {X_i(t, z): z in [0, 1]} for fixed t (a reinforcing stay, Appendix A plotA*/plotB*).

    Note: it lies in the leaflet only where t <= t_M(z); for t <= 5pi/3 that is every z in [0, 1].
    """
    z = np.linspace(0, 1, n)
    f = surface1 if surface == 1 else surface2
    return f(np.full_like(z, t), z)


# --- leaflet assembly & differential geometry ------------------------------------------------

def leaflet_grid(nt: int = 60, nz: int = 40, k: int = 0, z_min: float = 1e-3):
    """Return (P1, P2): point grids (nz, nt, 3) for the two cone patches of leaflet k (R_{2pi k/3}).

    Samples Omega with z in [z_min, 1]. The default z_min = 1e-3 TRUNCATES the closed domain (it omits the
    base arc of surface 1 and the apex row of surface 2, which is singular); pass z_min=0 for the closed Omega.
    """
    z = np.linspace(z_min, 1.0, nz)
    P1 = np.empty((nz, nt, 3))
    P2 = np.empty((nz, nt, 3))
    for i, zz in enumerate(z):
        t0, t1 = t_range_surface1(zz)
        P1[i] = surface1(np.linspace(t0, t1, nt), zz)
        s0, s1 = t_range_surface2(zz)
        P2[i] = surface2(np.linspace(s0, s1, nt), zz)
    ang = 2 * np.pi * k / 3
    return rot_ccw(P1, ang), rot_ccw(P2, ang)


def normal(surface: int, t, z, h: float = 1e-6):
    """Unit normal X_t x X_z by central differences (orientation: increasing t; see CONVENTIONS A-10)."""
    f = surface1 if surface == 1 else surface2
    Xt = (f(t + h, z) - f(t - h, z)) / (2 * h)
    Xz = (f(t, z + h) - f(t, z - h)) / (2 * h)
    n = np.cross(Xt, Xz)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def crease_angle_deg(z):
    """Angle between the tangent planes of surfaces 1 and 2 along the junction M(z), z in (0, 1].

    Unoriented (uses |n1 . n2|). 0 would mean a G1 (tangent-continuous) leaflet. Claim C-003.
    """
    z = np.atleast_1d(np.asarray(z, float))
    out = np.empty_like(z)
    for i, zz in enumerate(z):
        tm = t_M(zz)
        n1 = normal(1, tm, zz)
        n2 = normal(2, tm, zz)
        out[i] = np.degrees(np.arccos(np.clip(abs(n1 @ n2), -1, 1)))
    return out


def gaussian_curvature(surface: int, t, z, h: float = 1e-4):
    """Numerical Gaussian curvature via first/second fundamental forms (regular points only)."""
    f = surface1 if surface == 1 else surface2
    Xt = (f(t + h, z) - f(t - h, z)) / (2 * h)
    Xz = (f(t, z + h) - f(t, z - h)) / (2 * h)
    Xtt = (f(t + h, z) - 2 * f(t, z) + f(t - h, z)) / h**2
    Xzz = (f(t, z + h) - 2 * f(t, z) + f(t, z - h)) / h**2
    Xtz = (f(t + h, z + h) - f(t + h, z - h) - f(t - h, z + h) + f(t - h, z - h)) / (4 * h * h)
    n = np.cross(Xt, Xz)
    n = n / np.linalg.norm(n)
    E, F, G = Xt @ Xt, Xt @ Xz, Xz @ Xz
    L, M, N = Xtt @ n, Xtz @ n, Xzz @ n
    return (L * N - M * M) / (E * G - F * F)
