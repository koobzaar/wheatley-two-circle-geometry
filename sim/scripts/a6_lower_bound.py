"""A6: a lower bound for the total leaflet area over ALL profiles g in G, with a t-dependent Cauchy-Schwarz angle.

Setting (Section 6 of notes/technote/main.tex): theta = 2 pi / N, T(z) = t_M^theta(z), increasing from pi + theta (z = 0) to 2 pi
(z = 1) for N >= 3; c(t) = 1 + cos t; G = { g in C^1([0,1]) : g' >= 0, g(0) = 0, g(1) = 1 }.
    Atot_N(H, g) = N/2 int_0^1 int_pi^{T(g(s))} sqrt(4H^2 + g'(s)^2 c(t)^2) dt ds        (claim C-024).

Derivation.  For any measurable phi(t) in [0, pi/2]:
    sqrt(4H^2 + p^2 c^2) >= 2H cos(phi) + p c sin(phi)         (Cauchy-Schwarz, p >= 0, c >= 0).
Choose phi(t) with tan(phi(t)) = c(t) / (2H) on [pi, pi + theta] and phi = pi/2 on (pi + theta, 2 pi].  Then
  (i)  since T(g(s)) >= pi + theta for every s, the 2H cos(phi) part integrates over t in [pi, pi + theta] only, for every s;
  (ii) the p c sin(phi) part: int_0^1 g'(s) int_pi^{T(g(s))} c sin(phi) dt ds = int_0^1 Psi(T(z)) dz (substitution z = g(s),
       valid for g in C^1 non-decreasing with g(0) = 0, g(1) = 1), Psi(T) = int_pi^T c(t) sin(phi(t)) dt;
       and int_0^1 Psi(T(z)) dz = int_pi^{2pi} c(t) sin(phi(t)) mu(t) dt with mu(t) = |{z in [0,1] : T(z) >= t}| (Fubini).
  On [pi, pi + theta], mu = 1 and 2H cos + c sin = sqrt(4H^2 + c^2) at the chosen phi.  Hence, for every g in G,
    Atot_N(H, g) >= LB_N(H) = N/2 [ int_pi^{pi+theta} sqrt(4H^2 + c^2) dt + int_{pi+theta}^{2pi} c(t) mu(t) dt ].
  (With phi constant this reduces to the weaker Minkowski bound sqrt((2 pi H)^2 + pi^2), claim C-027.)
If LB_N(H) > target, no g in G reaches the target at height H ("impossible").
Run: uv run --project sim python sim/scripts/a6_lower_bound.py
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from wheatley import nleaflet as nl

TARGETS = {1.0: 8.680928480485932, 1.5: 12.305793191172732}   # N = 3, linear profile (a6_min_area.py, quad-certified)


def mu(N, t):
    th = 2 * np.pi / N
    if t <= np.pi + th:
        return 1.0
    if t >= 2 * np.pi:
        return 0.0
    return 1.0 - brentq(lambda z: float(nl.t_M(N, z)) - t, 0.0, 1.0, xtol=1e-15)


def lower_bound(N, H):
    th = 2 * np.pi / N
    a, ea = quad(lambda t: np.sqrt(4 * H * H + (1 + np.cos(t)) ** 2), np.pi, np.pi + th, epsabs=1e-13, epsrel=1e-13)
    b, eb = quad(lambda t: (1 + np.cos(t)) * mu(N, t), np.pi + th, 2 * np.pi, epsabs=1e-13, epsrel=1e-13, limit=200)
    return N / 2 * (a + b), N / 2 * (ea + eb)


if __name__ == "__main__":
    for H, target in TARGETS.items():
        for N in (3, 4, 5, 6, 7):
            lb, err = lower_bound(N, H)
            verdict = "impossible at this height" if lb - err > target else "not excluded"
            print(f"H = {H}, N = {N}: LB = {lb:.8f} (err {err:.1e}); target {target:.8f}: {verdict}")
