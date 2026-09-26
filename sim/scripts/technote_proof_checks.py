"""Symbolic checks of selected intermediate computations used in the written proofs of the technical note (sympy).

The Lean files certify part of the statements (Appendix A of the note); the note also gives readable proofs. This script
re-derives the algebraic steps those proofs use, so that the prose cannot drift from what was proved. The closed form
of k1 + k2 (Theorem 3.6) is derived separately in sim/scripts/junction_geodesic_curvature.py. Run:
    uv run --project sim python sim/scripts/technote_proof_checks.py        (also run by sim/tests/test_technote_proofs.py)
Numbering follows the proofs in notes/technote/main.tex.
"""
from __future__ import annotations

import sympy as sp

t, z, th, s = sp.symbols("t z theta s", real=True)
a_, b_, h, R, H, al, be = sp.symbols("a b h R H alpha beta", real=True)
b = z / 2
a = 1 - b
c, sn = sp.cos(t), sp.sin(t)


def rot(v, ang=th):
    C, S = sp.cos(ang), sp.sin(ang)
    return sp.Matrix([C * v[0] - S * v[1], S * v[0] + C * v[1], v[2]])


def zero(e):
    return sp.simplify(sp.expand_trig(sp.simplify(e))) == 0


def vzero(v):
    return all(zero(x) for x in v)


X1 = sp.Matrix([-b + a * c, a * sn, z])
X2 = rot(sp.Matrix([-a + b * c, b * sn, z]))
V1 = sp.Matrix([-1, 0, 2])
V2 = rot(sp.Matrix([-1, 0, 0]))
Q = rot(sp.Matrix([c, sn, 2]))
w1 = sp.Matrix([c + 1, sn, -2])
w2 = Q - V2
checks = {}

# 1. cone structure (Theorem 3.1): X1 = (1 - z/2)(cos t, sin t, 0) + (z/2) V1,  X2 = (1 - z/2) V2 + (z/2) Q
checks["1 cone X1"] = vzero(X1 - ((1 - z / 2) * sp.Matrix([c, sn, 0]) + z / 2 * V1))
checks["1 cone X2"] = vzero(X2 - ((1 - z / 2) * V2 + z / 2 * Q))
checks["1 cone map X1 = V1 + (1 - z/2) w1"] = vzero(X1 - (V1 + (1 - z / 2) * w1))
checks["1 cone map X2 = V2 + (z/2) w2"] = vzero(X2 - (V2 + z / 2 * w2))
checks["1 w2 = R_theta(cos t + 1, sin t, 2)"] = vzero(w2 - rot(sp.Matrix([c + 1, sn, 2])))

# 2. regularity: X_t x X_z = (alpha + beta z) beta (w' x w); closed forms of c_i = w_i' x w_i
c1 = w1.diff(t).cross(w1)
c2 = w2.diff(t).cross(w2)
checks["2 c1 = (-2cos t, -2sin t, -(1+cos t))"] = vzero(c1 - sp.Matrix([-2 * c, -2 * sn, -(1 + c)]))
checks["2 c2 = R_theta(2cos t, 2sin t, -(1+cos t))"] = vzero(c2 - rot(sp.Matrix([2 * c, 2 * sn, -(1 + c)])))
checks["2 |c1|^2 = 4 + (1+cos t)^2"] = zero(c1.dot(c1) - (4 + (1 + c) ** 2))
checks["2 |c2|^2 = 4 + (1+cos t)^2"] = zero(c2.dot(c2) - (4 + (1 + c) ** 2))
checks["2 X1t x X1z = (1 - z/2)(-1/2) c1"] = vzero(X1.diff(t).cross(X1.diff(z)) - (1 - z / 2) * sp.Rational(-1, 2) * c1)
checks["2 X2t x X2z = (z/2)(1/2) c2"] = vzero(X2.diff(t).cross(X2.diff(z)) - (z / 2) * sp.Rational(1, 2) * c2)

# 3. K = 0: X_zz = 0 and X_tz is parallel to X_t, so M = N = 0 in II
for name, X in (("X1", X1), ("X2", X2)):
    checks[f"3 {name}_zz = 0"] = vzero(X.diff(z, 2))
    checks[f"3 {name}_tz x {name}_t = 0"] = vzero(X.diff(t, z).cross(X.diff(t)))

# 4. flat patterns: |w1'| = 1, |w1|^2 = 6 + 2cos t, and the Lagrange identity behind r'^2 + r^2 phi'^2 = |w'|^2
checks["4 |w1'|^2 = 1"] = zero(w1.diff(t).dot(w1.diff(t)) - 1)
checks["4 |w1|^2 = 6 + 2cos t"] = zero(w1.dot(w1) - (6 + 2 * c))
checks["4 |w2|^2 = |w1|^2, |w2'| = 1"] = zero(w2.dot(w2) - w1.dot(w1)) and zero(w2.diff(t).dot(w2.diff(t)) - 1)
checks["4 Lagrange (w.w')^2 + |w' x w|^2 = |w|^2 |w'|^2"] = zero(
    w1.dot(w1.diff(t)) ** 2 + c1.dot(c1) - w1.dot(w1) * w1.diff(t).dot(w1.diff(t)))
r = sp.sqrt(6 + 2 * c)
checks["4 r r' = w.w'"] = zero(r * r.diff(t) - w1.dot(w1.diff(t)))

# 5. congruence T(x, y, z) = (R_theta(x, y), 2 - z): T(X1(t, z)) = X2(t, 2 - z), T(V1) = V2
def T(v):
    rv = rot(v)
    return sp.Matrix([rv[0], rv[1], 2 - v[2]])
checks["5 T(X1(t,z)) = X2(t,2-z)"] = vzero(T(X1) - X2.subs(z, 2 - z))
checks["5 T(V1) = V2"] = vzero(T(V1) - V2)
checks["5 stay length |X1(t,1) - X1(t,0)|^2 = (6 + 2cos t)/4"] = zero(
    (X1.subs(z, 1) - X1.subs(z, 0)).dot(X1.subs(z, 1) - X1.subs(z, 0)) - (6 + 2 * c) / 4)

# 6. fold angle: c1 . c2 = (1+cos t)^2 - 4 cos theta; at the junction 1 + cos t_M = (1 - cos theta)/Delta
checks["6 c1.c2"] = zero(c1.dot(c2) - ((1 + c) ** 2 - 4 * sp.cos(th)))
Dl = a_**2 + b_**2 - 2 * a_ * b_ * sp.cos(th)
cM = (2 * a_ * b_ - (a_**2 + b_**2) * sp.cos(th)) / Dl
sM = (b_**2 - a_**2) * sp.sin(th) / Dl
checks["6 cM^2 + sM^2 = 1 (a + b = 1)"] = zero((cM**2 + sM**2 - 1).subs(a_, 1 - b_))
checks["6 1 + cM = (1 - cos theta)/Delta (a + b = 1)"] = zero((1 + cM - (1 - sp.cos(th)) / Dl).subs(a_, 1 - b_))
bigP = sp.Matrix([a_ - 1 + a_ * sp.Symbol("C"), a_ * sp.Symbol("S"), 0])
smallP = rot(sp.Matrix([b_ - 1 + b_ * sp.Symbol("C"), b_ * sp.Symbol("S"), 0]))
diff_at_M = (bigP - smallP).subs({sp.Symbol("C"): cM, sp.Symbol("S"): sM}).subs(a_, 1 - b_)
checks["6 X1(t_M) = X2(t_M) (common parameter)"] = vzero(diff_at_M)
D = 1 - b_ + b_**2
X3 = (1 - sp.cos(2 * sp.pi / 3)) / Dl.subs({a_: 1 - b_, th: 2 * sp.pi / 3})
cd3 = (X3**2 - 4 * sp.cos(2 * sp.pi / 3)) / (X3**2 + 4)
checks["6 N = 3: cos delta = (8D^2 + 9)/(16D^2 + 9)"] = zero(cd3 - (8 * D**2 + 9) / (16 * D**2 + 9))
u = sp.symbols("u", positive=True)
f = (8 * u + 9) / (16 * u + 9)
checks["6 (8u+9)/(16u+9) decreasing"] = zero(sp.diff(f, u) + 72 / (16 * u + 9) ** 2)
checks["6 bounds: f(9/16) = 3/4, f(1) = 17/25"] = f.subs(u, sp.Rational(9, 16)) == sp.Rational(3, 4) and f.subs(u, 1) == sp.Rational(17, 25)
checks["6 free edge: cos delta = (1 - cos theta)/2"] = zero(((4 - 4 * sp.cos(th)) / 8) - (1 - sp.cos(th)) / 2)
w1h = sp.Matrix([c + 1, sn, -2 * h])
w2h = rot(sp.Matrix([c + 1, sn, 2 * h]))
c1h, c2h = w1h.diff(t).cross(w1h), w2h.diff(t).cross(w2h)
checks["6 vertical scale: c1h.c2h = (1+cos t)^2 - 4h^2 cos theta"] = zero(c1h.dot(c2h) - ((1 + c) ** 2 - 4 * h**2 * sp.cos(th)))
checks["6 vertical scale: |c1h|^2 = 4h^2 + (1+cos t)^2"] = zero(c1h.dot(c1h) - (4 * h**2 + (1 + c) ** 2))
Xh = 3 / (2 * D)
checks["6 N = 3, scale h: (8h^2D^2+9)/(16h^2D^2+9)"] = zero((Xh**2 + 2 * h**2) / (Xh**2 + 4 * h**2) - (8 * h**2 * D**2 + 9) / (16 * h**2 * D**2 + 9))

# 7. a + b = 1: with e = (1, 0), u = (cos t, sin t), v = e + u:  |a v - e|^2 = 1 - a(1 - a)|v|^2
e2 = sp.Matrix([1, 0]); v2 = sp.Matrix([1 + c, sn])
checks["7 |a v - e|^2 = 1 - a(1-a)|v|^2"] = zero((a_ * v2 - e2).dot(a_ * v2 - e2) - (1 - a_ * (1 - a_) * v2.dot(v2)))
checks["7 a(1-a) - b(1-b) = (a-b)(1-a-b)"] = sp.expand(a_ * (1 - a_) - b_ * (1 - b_) - (a_ - b_) * (1 - a_ - b_)) == 0

# 8. no common tangent plane: (V2 - V1).c1(t1) = 2(1 + cos(t1 - theta))
t1 = sp.symbols("t1", real=True)
checks["8 (V2 - V1).c1 = 2(1 + cos(t1 - theta))"] = zero((V2 - V1).dot(c1.subs(t, t1)) - 2 * (1 + sp.cos(t1 - th)))
checks["8 stays lie in the plane through V1 with normal c1"] = zero((X1 - V1).dot(c1))
checks["8 stays lie in the plane through V2 with normal c2"] = zero((X2 - V2).dot(c2))

# 9. area density (0 <= z <= 2): |X1t x X1z| + |X2t x X2z| = (1/2) sqrt(4 + (1+cos t)^2)
checks["9 (1 - z/2)/2 + (z/2)/2 = 1/2"] = sp.simplify((1 - z / 2) / 2 + (z / 2) / 2 - sp.Rational(1, 2)) == 0

# 10. height H, radius R, profile g: area elements (Section 6)
g = sp.Function("g")(s)
bz = g / 2
Y1 = sp.Matrix([R * (-bz + (1 - bz) * c), R * (1 - bz) * sn, H * s])
Y2 = rot(sp.Matrix([R * (-(1 - bz) + bz * c), R * bz * sn, H * s]))
n1 = Y1.diff(t).cross(Y1.diff(s)); n2 = Y2.diff(t).cross(Y2.diff(s))
target = (4 * H**2 + R**2 * g.diff(s) ** 2 * (1 + c) ** 2)
checks["10 |Y1t x Y1s|^2 = (R(1-g/2)/2)^2 (4H^2 + R^2 g'^2 (1+cos t)^2)"] = zero(n1.dot(n1) - (R * (1 - bz) / 2) ** 2 * target)
checks["10 |Y2t x Y2s|^2 = (R g/4)^2 (4H^2 + R^2 g'^2 (1+cos t)^2)"] = zero(n2.dot(n2) - (R * bz / 2) ** 2 * target)

# 11. projected Jacobians (covering proposition): piece 1 by (t, z), piece 2 oppositely
J1 = sp.Matrix([[X1[0].diff(t), X1[0].diff(z)], [X1[1].diff(t), X1[1].diff(z)]]).det()
J2 = sp.Matrix([[X2[0].diff(t), X2[0].diff(z)], [X2[1].diff(t), X2[1].diff(z)]]).det()
checks["11 J1 = a(1+cos t)/2"] = zero(J1 - a * (1 + c) / 2)
checks["11 -J2 = b(1+cos t)/2"] = zero(-J2 - b * (1 + c) / 2)

# 12. second fundamental form of the cones: w1''.c1 = 2, w2''.c2 = -2
checks["12 w1''.c1 = 2"] = zero(w1.diff(t, 2).dot(c1) - 2)
checks["12 w2''.c2 = -2"] = zero(w2.diff(t, 2).dot(c2) + 2)

# 13. coaptation: the circles of X1 (centre (-b, 0), radius a) and of the unrotated small circle (centre (-a, 0),
#     radius b) have centre distance a - b = difference of radii: internally tangent (at E) for a != b
checks["13 |(-b) - (-a)| = a - b"] = sp.simplify((a - b) - (-b - (-a))) == 0


def run():
    bad = [k for k, ok in checks.items() if not ok]
    for k, ok in checks.items():
        print(("ok   " if ok else "FAIL ") + k)
    print(f"{len(checks) - len(bad)}/{len(checks)} checks passed")
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if run() else 0)
