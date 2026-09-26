"""A6 (exploration): total leaflet area N*A_N(H, g) for the two-circle family with height H and vertical profile g.

Area density (sympy, this session): |X1_t x X1_s| + |X2_t x X2_s| = (1/2) sqrt(4 H^2 + g'(s)^2 (1 + cos t)^2)
for 0 <= g <= 2, where the section at physical height H*s uses z = g(s) (b = g/2, a = 1 - b) and theta = 2*pi/N.
Profile family g(s) = s^m (McKee 2021 suggests z -> z^m). For each N, find m such that N*A_N(H, s^m) equals the
N = 3, m = 1 value at the same H.
Run: uv run --project sim python sim/scripts/a6_area_profile.py
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from wheatley import nleaflet as nl


def area(N: int, H: float, m: float) -> float:
    """One leaflet, profile g(s) = s^m, g'(s) = m s^(m-1)."""
    def inner(s):
        gp = m * s ** (m - 1) if s > 0 else (m if m == 1 else (0.0 if m > 1 else np.inf))
        tM = float(nl.t_M(N, s ** m))
        return quad(lambda t: np.sqrt(4 * H * H + gp * gp * (1 + np.cos(t)) ** 2), np.pi, tM, epsabs=1e-11)[0]
    return 0.5 * quad(inner, 0.0, 1.0, epsabs=1e-10, limit=200)[0]


def roots_in_m(N, H, ref, grid=np.geomspace(0.1, 20, 61)):
    f = lambda m: N * area(N, H, m) - ref
    v = np.array([f(m) for m in grid])
    out = []
    for k in range(len(grid) - 1):
        if v[k] == 0:
            out.append(grid[k])
        elif v[k] * v[k + 1] < 0:
            out.append(brentq(f, grid[k], grid[k + 1], xtol=1e-7))
    return out, v.min() + ref, v.max() + ref


def height_for_area(N, ref):
    """Linear profile: height H_N with N*A_N(H_N, s) = ref (N*A_N is increasing in H)."""
    return brentq(lambda H: N * area(N, H, 1.0) - ref, 0.05, 20.0, xtol=1e-9)


if __name__ == "__main__":
    for H in (1.0, 1.5):
        ref = 3 * area(3, H, 1.0)
        print(f"H = {H}: target = total area of N = 3 with linear profile = {ref:.5f}")
        for N in (2, 3, 4, 5, 6):
            rts, lo, hi = roots_in_m(N, H, ref)
            sup2 = f"; N=2 sup (wall limit) = {np.pi * (2 * H + 1):.5f}" if N == 2 else ""
            print(f"  N = {N}: linear {N * area(N, H, 1.0):.5f}; m with equal area: "
                  f"{[round(r, 4) for r in rts]}; range over m in [0.1, 20]: [{lo:.4f}, {hi:.4f}]{sup2}")
        print("  linear profile, height needed for equal area:",
              {N: round(height_for_area(N, ref), 5) for N in (2, 3, 4, 5, 6)})
