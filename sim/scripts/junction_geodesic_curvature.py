"""C-011: geodesic curvature of the junction curve M(z) on the two cone patches (exact, sympy).

For a curve gamma on a piecewise-smooth surface made of two patches glued along gamma, the
intrinsic metric is flat across gamma iff kg1 + kg2 = 0, where kg_i = gamma'' . nu_i (arclength)
and nu_i is the unit conormal of patch i pointing INTO patch i (Gauss-Bonnet; e.g. the curvature
measure of a piecewise flat/ smooth surface).  For the leaflet both patches lie on the side
t <= t_M(z) of the junction, so nu_i is the component of -dX_i/dt orthogonal to gamma'.

Conventions: notes/technote/main.tex (Section 1) (b = z/2, t_M from p.5-6, M from eq. (33)).
Run: uv run --project sim python sim/scripts/junction_geodesic_curvature.py
"""
import sympy as sp

b = sp.symbols('b', positive=True)
s3 = sp.sqrt(3)
D = 1 - b + b**2
cM = (1 + 2*b - 2*b**2) / (2*D)          # cos t_M  (p.5; Lean cos_tM)
sM = -s3*(1 - 2*b) / (2*D)               # sin t_M  (Lean sin_tM)
z = 2*b
# M(z), eq. (33), as a function of b; d/dz = (1/2) d/db
M = sp.Matrix([(1 - b - 2*b**2)/(2*D), -s3*(1 - b)*(1 - 2*b)/(2*D), z])
dM = M.diff(b)/2
ddM = dM.diff(b)/2
# dX_i/dt at t = t_M
X1t = sp.Matrix([-(1 - b)*sM, (1 - b)*cM, 0])
X2t = sp.Matrix([-b*(-sM + s3*cM)/2 * 1, b*(-s3*sM - cM)/2, 0])
# (surf2)_t = (1/2)*(-b*(-sin t + sqrt3 cos t), b*(-sqrt3 sin t - cos t)) ; check below
t = sp.symbols('t')
X2 = sp.Matrix([sp.Rational(1, 2)*((1 - b) - b*(sp.cos(t) + s3*sp.sin(t))),
                sp.Rational(1, 2)*(-s3*(1 - b) + b*(s3*sp.cos(t) - sp.sin(t))), 2*b])
X2t_chk = X2.diff(t).subs({sp.cos(t): cM, sp.sin(t): sM})
X2t = X2t_chk


def kg(Xt):
    sp2 = dM.dot(dM)
    # curvature vector for a non-arclength parametrisation: (ddM - (ddM.T)T)/|dM|^2
    T = dM / sp.sqrt(sp2)
    kv = (ddM - ddM.dot(T)*T) / sp2
    nu = -Xt
    nu = nu - nu.dot(T)*T
    return kv.dot(nu) / sp.sqrt(nu.dot(nu))


def main():
    k1 = kg(X1t)
    k2 = kg(X2t)
    for bv in [sp.Rational(1, 20), sp.Rational(1, 8), sp.Rational(1, 4), sp.Rational(3, 8), sp.Rational(9, 20)]:
        v1 = sp.nsimplify(sp.simplify(k1.subs(b, bv)))
        v2 = sp.nsimplify(sp.simplify(k2.subs(b, bv)))
        tot = sp.simplify(k1.subs(b, bv) + k2.subs(b, bv))
        print(f"z={float(2*bv):.3f}  kg1={float(k1.subs(b,bv)):+.10f}  kg2={float(k2.subs(b,bv)):+.10f}"
              f"  sum={float(tot):+.10f}")
    # exact value at z = 1/2 (b = 1/4)
    ex = sp.radsimp(sp.simplify(k1.subs(b, sp.Rational(1, 4)) + k2.subs(b, sp.Rational(1, 4))))
    print("exact sum at z=1/2:", ex, "=", sp.N(ex, 15))
    print("kg1 at b=1/2:", sp.N(k1.subs(b, sp.Rational(1, 2)), 15), " kg2:", sp.N(k2.subs(b, sp.Rational(1, 2)), 15))
    # integrated angle defect along the junction: int_0^1 (kg1+kg2) |M'(z)| dz
    import mpmath as mp
    f = sp.lambdify(b, (k1 + k2) * sp.sqrt(dM.dot(dM)) / 1, 'mpmath')
    mp.mp.dps = 20
    I = mp.quad(lambda bb: 2*f(bb), [0, sp.Rational(1, 2)])   # dz = 2 db
    print("int_M (kg1+kg2) ds =", I)


def closed_form_check():
    """Closed form:
    kg1 + kg2 = 4 sqrt3 (z - 1) Dz^2 / ( sqrt(Dz^2 + 9) (Dz^2 + 12)^(3/2) ),  Dz = z^2 - 2z + 4 = 4 D(b).
    Checked here numerically at 40-digit precision on the 49 rationals b = 0.01, ..., 0.49, and exactly
    (as algebraic numbers) at z = 1/2 (b = 1/4). No general symbolic identity is attempted here.
    Raises AssertionError if either check fails."""
    zz = 2*b
    Dz = zz**2 - 2*zz + 4
    cf = 4*s3*(zz - 1)*Dz**2 / (sp.sqrt(Dz**2 + 9) * (Dz**2 + 12)**sp.Rational(3, 2))
    tot = kg(X1t) + kg(X2t)
    worst = 0
    for k in range(1, 50):
        bv = sp.Rational(k, 100)
        d = abs(sp.N(tot.subs(b, bv) - cf.subs(b, bv), 40))
        worst = max(worst, d)
    print("closed form: max |difference| on b = 0.01..0.49:", worst)
    # exact check at z = 1/2 (b = 1/4): both sides are algebraic numbers; compare them exactly
    ex_l = sp.radsimp(sp.nsimplify(sp.simplify(tot.subs(b, sp.Rational(1, 4)))))
    ex_r = sp.radsimp(sp.simplify(cf.subs(b, sp.Rational(1, 4))))
    exact_equal = sp.simplify(ex_l - ex_r) == 0
    print("exact at z=1/2:", ex_l, "vs", ex_r, "equal:", exact_equal)
    assert worst < sp.Float('1e-30', 40), worst
    assert exact_equal
    return worst


if __name__ == '__main__':
    main()
    closed_form_check()
