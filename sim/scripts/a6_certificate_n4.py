"""A6: interval-arithmetic certificate that N = 4 at H = 1 cannot reach the area of the N = 3 linear design (claim C-030),
and a certified version of the N = 5 bound of claim C-029.
The closed forms used for T(z) - pi,
pi - 2 atan(1 - z) (N = 4) and pi - 2 atan((1 - z)/sqrt 3) (N = 3), were checked symbolically (cos and sin of both sides agree
with (cMθ, sMθ) of C-012; both lie in [pi + theta, 2 pi]).
Run: uv run --project sim python sim/scripts/a6_certificate_n4.py

Finite Cauchy certificate, independent of project imports and optimizer script.

For N=4,H=1,k=31/5, use left rectangles in z and t. On each z strip,
T(z)>=T_i and c(t)=1+cos(t) increases on [pi,2pi]. Thus
Phi(z,u) >= (2 L_i/J) sum_j hypot(2u,c_ij).
For positive v_i and alpha=2v/hypot(2v,c), beta=c/hypot(2v,c),
the RHS >= a_i*u+b_i, with a_i=4 L_i/J sum alpha and
b_i=2 L_i/J sum c*beta. If a_i>=k for every i, weak duality gives
area>=k+sum b_i/M. A finite interval evaluation verifies this condition.
The target is upper-bounded by right rectangles in z,t (both monotone).
No quadrature error estimates or optimizer convergence assumptions used.
"""
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from mpmath import iv
iv.dps=30
M=J=200
k=iv.mpf(31)/5
lower=k
min_slack=iv.mpf('100')
for i in range(M):
    Li=iv.pi-2*iv.atan2(1-iv.mpf(i)/M,iv.mpf(1))
    L=float(Li.mid)
    c=1+np.cos(np.pi+L*np.arange(J)/J)
    v=brentq(lambda v:4*L*np.mean(2*v/np.hypot(2*v,c))-6.2,1e-12,8)
    vi=iv.mpf(str(v*1.000001))
    slope=iv.mpf(0); intercept=iv.mpf(0)
    for j in range(J):
        ci=1+iv.cos(iv.pi+Li*j/J)
        den=iv.sqrt(4*vi*vi+ci*ci)
        slope+=2*vi/den
        intercept+=ci*ci/den
    slope*=4*Li/J
    intercept*=2*Li/J
    assert slope.a>k.b, (i,slope,k)
    if (slope-k).a<min_slack.a:min_slack=slope-k
    lower+=intercept/M

# N=3,H=1,g(s)=s target. T(z) monotone; sqrt(4+c(t)^2) monotone.
upper=iv.mpf(0)
for i in range(1,M+1):
    Li=iv.pi-2*iv.atan2((1-iv.mpf(i)/M)/iv.sqrt(3),iv.mpf(1))
    row=iv.mpf(0)
    for j in range(1,J+1):
        ci=1+iv.cos(iv.pi+Li*j/J)
        row+=iv.sqrt(4+ci*ci)
    upper+=iv.mpf(3)/2*Li*row/(M*J)

assert lower.a>iv.mpf('8.79').b
assert upper.b<iv.mpf('8.72').a
# C-029: the simplified integrand decreases, hence right rectangles are lower.
lb5=iv.pi
theta=2*iv.pi/5
for j in range(1,1001):
    c=1-iv.cos(theta*j/1000)
    lb5+=iv.mpf(5)/2*theta/1000*(iv.sqrt(4+c*c)-c)
assert lb5.a>upper.b
out=f'M={M}, J={J}\nminimum slope slack={min_slack}\nlower certificate={lower}\ntarget upper certificate={upper}\nC029 N5 lower certificate={lb5}\nPROVED: all g in G have N4,H1 area > 8.79 > 8.72 > target.\n'
print(out)
(Path(__file__).parent / '..' / 'out' / 'a6' / 'certificate_n4.txt').write_text(out)
