"""Audit of McKee et al. 2026, eqs. (25)-(29): how (27)-(28) arise and what (28) should read.

Hypothesis H1: the paper equates the parametrisation of the large circle (13)-(14) and of the
rotated small circle (25)-(26) *with the same parameter t*, which gives two linear equations
in (cos t, sin t).  Printed: cos t = -(2(a^2+ab+b^2) - 3(a+b)) / (2(a^2+ab+b^2)),
sin t = -sqrt(a-b) / (2(a^2+ab+b^2)).
"""
import sympy as sp

a, b = sp.symbols("a b", positive=True)
c, s = sp.symbols("c s")  # cos t, sin t
r3 = sp.sqrt(3)

big = (-(1 - a) + a * c, a * s)                                   # (13)-(14)
small = (sp.Rational(1, 2) * (1 - b - b * (c + r3 * s)),          # (25)
         sp.Rational(1, 2) * (-r3 * (1 - b) + b * (r3 * c - s)))  # (26)
sol = sp.solve([big[0] - small[0], big[1] - small[1]], [c, s], dict=True)[0]
Q = a**2 + a * b + b**2
print("H1 cos t =", sp.factor(sol[c]))
print("H1 sin t =", sp.factor(sol[s]))
print("printed (27) matches H1:", sp.simplify(sol[c] + (2 * Q - 3 * (a + b)) / (2 * Q)) == 0)
print("sqrt3(a-b) variant of (28) matches H1:", sp.simplify(sol[s] + r3 * (a - b) / (2 * Q)) == 0)
print("printed sqrt(a-b) variant matches H1:", sp.simplify(sol[s] + sp.sqrt(a - b) / (2 * Q)) == 0)
rel = sp.factor(sp.together(sol[c] ** 2 + sol[s] ** 2 - 1))
print("cos^2+sin^2-1 (H1) factors:", rel)
print("roots in a:", sp.solve(sp.numer(rel), a))
# with the printed (28):
relp = sp.factor(sp.together(sol[c] ** 2 + (sp.sqrt(a - b) / (2 * Q)) ** 2 - 1))
print("with printed (28), numerator:", sp.factor(sp.numer(relp)))

# --- residual of the *printed* (28) on the family a = 1 - b
bb = sp.symbols("b", real=True)
Qb = (1 - bb) ** 2 + (1 - bb) * bb + bb ** 2
cb = -(2 * Qb - 3) / (2 * Qb)
res_printed = sp.factor(sp.together(cb ** 2 + (sp.sqrt(1 - 2 * bb) / (2 * Qb)) ** 2 - 1))
print("printed (28) on a=1-b: cos^2+sin^2-1 =", res_printed)
print("zeros in [0,1/2]:", [r for r in sp.solve(sp.numer(res_printed), bb) if 0 <= r <= sp.Rational(1, 2)])
print("value at b=1/4:", res_printed.subs(bb, sp.Rational(1, 4)))
