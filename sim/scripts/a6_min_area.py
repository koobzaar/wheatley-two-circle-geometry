"""A6: can N = 2, 3, 4 leaflets have the same height H AND the same total area as the N = 3 linear design?

Formulation (Section 6 of notes/technote/main.tex): ring radius 1, theta = 2 pi / N, height H > 0, profile g in
    G = { g in C^1([0,1]) : g' >= 0, g(0) = 0, g(1) = 1 }   (so 0 <= g <= 1).
Total area (density: claim C-024, valid for 0 <= g <= 1 and theta = 2 pi / N):
    Atot_N(H, g) = N/2 int_0^1 int_pi^{T(g(s))} sqrt(4 H^2 + g'(s)^2 (1 + cos t)^2) dt ds,   T = t_M^theta.

Parametrisation (every parameter vector gives a g in G exactly): g' is the continuous piecewise-linear function with
positive nodal values q_k = exp(x_k) on a uniform grid of K cells, divided by its integral; g is its (piecewise
quadratic) primitive.  So g' >= 0 is continuous, g(0) = 0, g(1) = 1.

Conclusions this script is allowed to print (rules in PROBLEM.md):
  * "attainable": an explicit g in G whose area, re-evaluated with adaptive quadrature (scipy.quad), matches the target
    to 1e-8.  It is found by the intermediate value theorem on a segment of G (G is convex, the area is continuous).
  * "impossible": only if the Minkowski lower bound  Atot >= sqrt((2 pi H)^2 + (N W_N)^2) exceeds the target (NOTE: N W_N = pi
    for every N, claim C-027, so this bound never excludes the N = 3 linear target; kept only as a sanity check), where
    W_N = 1/2 int_0^1 int_pi^{T(z)} (1 + cos t) dt dz.  (Proof: sqrt(u^2 + v^2) >= u cos(phi) + v sin(phi); integrate;
    int_0^1 (T(g(s)) - pi) ds >= theta because T is increasing from pi + theta; change variables z = g(s) in the v-term;
    maximise over phi.)
  * "inconclusive": otherwise.  The numerical minimum over the parametrised family is only an UPPER bound for inf over G.
Run: uv run --project sim python sim/scripts/a6_min_area.py
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize

from wheatley import nleaflet as nl

NS, NT = 8, 64  # Gauss-Legendre nodes per s-cell and in t
XS, WS = np.polynomial.legendre.leggauss(NS)
XT, WT = np.polynomial.legendre.leggauss(NT)


def profile(x, K):
    """Nodal values -> (g, g') as callables; g' piecewise linear, positive, unit integral."""
    q = np.exp(x - np.max(x))
    q = q / (np.sum(q[:-1] + q[1:]) / (2 * K))
    nodes = np.linspace(0.0, 1.0, K + 1)
    cum = np.concatenate([[0.0], np.cumsum((q[:-1] + q[1:]) / (2 * K))])

    def gp(s):
        return np.interp(s, nodes, q)

    def g(s):
        s = np.asarray(s, dtype=float)
        k = np.clip((s * K).astype(int), 0, K - 1)
        u = s - nodes[k]
        return cum[k] + q[k] * u + (q[k + 1] - q[k]) * K * u * u / 2
    return g, gp


def total_area(N, H, g, gp, K):
    """Gauss quadrature, K cells in s (exact cell boundaries at the kinks of g')."""
    a = np.linspace(0.0, 1.0, K + 1)
    s = ((a[:-1, None] + a[1:, None]) / 2 + (XS[None, :] / 2) / K).ravel()
    ws = np.tile(WS / (2 * K), K)
    z, p = np.clip(g(s), 0.0, 1.0), gp(s)
    T = np.asarray(nl.t_M(N, z), dtype=float)
    tt = np.pi + (T[:, None] - np.pi) * (XT[None, :] + 1) / 2
    f = np.sqrt(4 * H * H + (p[:, None] * (1 + np.cos(tt))) ** 2)
    inner = (T - np.pi) / 2 * (f * WT[None, :]).sum(axis=1)
    return N * 0.5 * np.sum(ws * inner)


def total_area_quad(N, H, g, gp, K):
    """Independent adaptive evaluation with error control: the outer integral is split at the K
    kinks of g', both levels use epsabs = epsrel = 1e-13, and the returned error bound adds the inner and outer estimates.
    Raises if scipy reports an IntegrationWarning."""
    import warnings
    from scipy.integrate import IntegrationWarning
    inner_err = [0.0]

    def inner(s):
        T = float(nl.t_M(N, float(g(s))))
        v, e = quad(lambda t: np.sqrt(4 * H * H + (gp(s) * (1 + np.cos(t))) ** 2), np.pi, T,
                    epsabs=1e-13, epsrel=1e-13, limit=200)
        inner_err[0] = max(inner_err[0], e)
        return v
    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        total, err = 0.0, 0.0
        for a, b in zip(np.linspace(0, 1, K + 1)[:-1], np.linspace(0, 1, K + 1)[1:]):
            v, e = quad(inner, a, b, epsabs=1e-13, epsrel=1e-13, limit=200)
            total, err = total + v, err + e
    return N * 0.5 * total, N * 0.5 * (err + inner_err[0])


def lower_bound(N, H):
    W = 0.5 * quad(lambda z: (float(nl.t_M(N, z)) - np.pi) + np.sin(float(nl.t_M(N, z))), 0.0, 1.0, epsabs=1e-12)[0]
    return np.hypot(2 * np.pi * H, N * W)


def minimise(N, H, K):
    fun = lambda x: total_area(N, H, *profile(x, K), K)
    r = minimize(fun, np.zeros(K + 1), method="L-BFGS-B", options={"maxiter": 5000})
    return r.fun, r.x


def study(N, H, target, K=40, target_err=0.0):
    lin = np.zeros(K + 1)                        # all q equal -> g(s) = s
    amin, xmin = minimise(N, H, K)
    lb = lower_bound(N, H)
    a_lin = total_area(N, H, *profile(lin, K), K)
    verdict, xstar = "inconclusive", None
    if lb > target:
        verdict = "impossible (lower bound > target)"
    else:
        # segment in G between the numerical minimiser and a steep profile (or the linear one) that brackets the target
        steep = np.where(np.arange(K + 1) <= 1, 8.0, 0.0) if a_lin < target else lin   # g rises at s ~ 0
        for other in (lin, steep):
            A0 = total_area(N, H, *profile(xmin, K), K)
            A1 = total_area(N, H, *profile(other, K), K)
            if (A0 - target) * (A1 - target) <= 0:
                # convex combination of the two g' (not of x): g_lam = (1-lam) g_min + lam g_other stays in G
                g0, gp0 = profile(xmin, K)
                g1, gp1 = profile(other, K)
                comb = lambda lam: (lambda s: (1 - lam) * g0(s) + lam * g1(s), lambda s: (1 - lam) * gp0(s) + lam * gp1(s))
                lam = brentq(lambda l: total_area(N, H, *comb(l), K) - target, 0.0, 1.0, xtol=1e-12)
                g, gp = comb(lam)
                A_q, e_q = total_area_quad(N, H, g, gp, K)
                witness = dict(N=N, H=H, K=K, x_min=xmin.tolist(), x_other=other.tolist(), lam=repr(lam),
                               area_quad=repr(A_q), area_err_estimate=repr(e_q), target=repr(target),
                               target_err_estimate=repr(target_err))
                if abs(A_q - target) + e_q + target_err < 1e-8:
                    verdict, xstar = f"attainable (lambda = {lam!r})", witness
                else:
                    verdict, xstar = "inconclusive (quad re-check failed)", witness
                break
    return dict(N=N, H=H, linear=a_lin, min_upper=amin, lower=lb, verdict=verdict, g=xstar)


if __name__ == "__main__":
    import json
    import os
    out = os.path.join(os.path.dirname(__file__), "..", "out", "a6")
    os.makedirs(out, exist_ok=True)
    for H in (1.0, 1.5):
        lin = profile(np.zeros(41), 40)
        target, terr = total_area_quad(3, H, *lin, 40)          # certified target (N = 3, linear)
        print(f"H = {H}: target (N = 3, linear) = {target!r} (quad error bound {terr:.1e})")
        for N in (2, 3, 4):
            r = study(N, H, target, target_err=terr)
            print(f"  N = {N}: linear {r['linear']:.8f}; min over family (upper bound of inf) {r['min_upper']:.8f}; "
                  f"lower bound {r['lower']:.8f} (useless: N W_N = pi); {r['verdict']}")
            if r["g"] is not None:
                f = os.path.join(out, f"witness_N{N}_H{H}.json")
                with open(f, "w") as fh:
                    json.dump(r["g"], fh, indent=1)
                print(f"    witness saved: {os.path.relpath(f)}")
