"""A6: the equal-area family G_N (criterion: Section 6 of the note).

Criterion: same total leaflet area S for every N; the paper's linear profile g(s) = s; GOA is a response, not imposed.
Scaling law (claim C-031): Area(N, R, H) = R^2 * A_N(H / R), with A_N(h) = N/2 int_0^1 int_pi^{t_M(z)} sqrt(4h^2 + c^2) dt dz,
c = 1 + cos t (claim C-024 with g' = 1).  So the shape depends only on (N, h = H/R); two ways of fixing the scale:
  (R fixed)  R = 1, h_N solves A_N(h) = S.
             A_N is continuous and strictly increasing in h (the integrand is), A_N(h) -> N W_N = pi as h -> 0 (claim C-027:
             projected area) and -> infinity as h -> infinity; so a unique h_N exists iff S > pi.
  (H fixed)  H = H0, R_N solves f(R) = R^2 A_N(H0 / R) = S.
             f'(R) = R (2 A_N(h) - h A_N'(h)) with h = H0/R, and h A_N'(h) = N/2 int int 4h^2 / sqrt(4h^2 + c^2) <= A_N(h), so
             f' >= R A_N(h) > 0: f is strictly increasing; A_N(h) <= pi + N pi h (since int_0^1 (t_M(z) - pi) dz <= pi), so
             f <= pi R^2 + N pi H0 R -> 0 as R -> 0, and f >= pi R^2 -> infinity; a unique R_N exists for every S > 0.
  The shape family is indexed by (N, h); the two constraints select DIFFERENT shapes of it: h_N (R fixed)
  versus H0 / R_N (H fixed). Both are reported.
Reported per N (dimensionless, depend on (N, h) only): h, the fold angle along the junction (claim C-021 formula
  cos delta = ((1 + cos t_M)^2 - 4 h^2 cos theta) / (4 h^2 + (1 + cos t_M)^2), min over 0 < z <= 1 and supremum = limit z -> 0+, evaluated at z = 0 in the formula), and the scale (R or H).
  Naming: Atot = total area of the N leaflets (C-024's A_N is per leaflet).
Reference S: N = 3, R = 1, linear profile, h0 in {1, 1.5}.
Run: uv run --project sim python sim/scripts/a6_equal_area_family.py
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from wheatley import nleaflet as nl


def Atot(N, h):
    inner = lambda z: quad(lambda t: np.sqrt(4 * h * h + (1 + np.cos(t)) ** 2), np.pi, float(nl.t_M(N, z)),
                           epsabs=1e-13, epsrel=1e-13)[0]
    return N / 2 * quad(inner, 0.0, 1.0, epsabs=1e-12, epsrel=1e-12, limit=200)[0]


def fold_deg(N, h, z):
    th = 2 * np.pi / N
    X = 1 + np.cos(float(nl.t_M(N, z)))
    return np.degrees(np.arccos((X * X - 4 * h * h * np.cos(th)) / (4 * h * h + X * X)))


if __name__ == "__main__":
    zs = np.concatenate([[0.0], np.linspace(1e-3, 1.0, 400)])   # z = 0: the limit z -> 0+ (supremum, not attained)
    for h0 in (1.0, 1.5):
        S = Atot(3, h0)
        print(f"reference: N = 3, R = 1, h = {h0}: total area S = {S:.8f}")
        for N in range(2, 8):
            hR = brentq(lambda h: Atot(N, h) - S, 1e-6, 50.0, xtol=1e-12)            # R fixed = 1
            RH = brentq(lambda R: R * R * Atot(N, h0 / R) - S, 1e-3, 50.0, xtol=1e-12)  # H fixed = h0
            hH = h0 / RH
            fr = [fold_deg(N, hR, z) for z in zs]
            fh = [fold_deg(N, hH, z) for z in zs]
            print(f"  N = {N}: R fixed -> H = {hR:.5f} (h = {hR:.5f}), fold [{min(fr):.2f}, {max(fr):.2f}] deg | "
                  f"H fixed -> R = {RH:.5f} (h = {hH:.5f}), fold [{min(fh):.2f}, {max(fh):.2f}] deg")
