"""C-016 (numeric part): the N-leaflet valve has no contacts other than the designed ones.

At each height z (25 values in [0.02, 0.98]) the cross-section of leaflet k consists of two arcs
(surface 1 on t in [pi, t_M], surface 2 on the same range).  We compute the minimum distance
between every pair of arcs, excluding only the designed contacts:
  - the two arcs of the same leaflet share the junction point M (last 10% of samples removed);
  - arc 2 of leaflet k and arc 1 of leaflet k+1 share the commissure point (first 10% removed).
All other pairs (non-neighbouring leaflets, etc.) are compared in full.  Output: the minimum
separation per N.  It tends to 0 as z -> 1, where arc 2 of leaflet k and arc 1 of leaflet k+1
coincide (C-013), and also as z -> 0, where the junction M(z) approaches the commissure R_theta E,
which lies on arc 1 of the next leaflet at every height.
Run: uv run --project sim python sim/scripts/nleaflet_separation.py
"""
import numpy as np

from wheatley import geometry as g
from wheatley import nleaflet as nl


def arcs(N, z, n=400):
    t = np.linspace(np.pi, float(nl.t_M(N, z)), n)
    A1 = g.surface1(t, z)[:, :2]
    A2 = nl.surface2(N, t, z)[:, :2]
    out = []
    for k in range(N):
        ang = 2 * np.pi * k / N
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        out.append((k, 1, A1 @ R.T))
        out.append((k, 2, A2 @ R.T))
    return out


def mind(P, Q):
    return np.min(np.linalg.norm(P[:, None] - Q[None], axis=-1))


def min_separation(N, zs=np.linspace(0.02, 0.98, 25)):
    worst, info = np.inf, None
    for z in zs:
        A = arcs(N, z)
        cut = 40
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                ki, si, P = A[i]
                kj, sj, Q = A[j]
                if ki == kj:
                    d = mind(P[:-cut], Q[:-cut])
                elif (si, sj) in ((2, 1), (1, 2)) and (
                        (si == 2 and kj == (ki + 1) % N) or (sj == 2 and ki == (kj + 1) % N)):
                    d = mind(P[cut:], Q[cut:])
                else:
                    d = mind(P, Q)
                if d < worst:
                    worst, info = d, (float(z), (ki, si), (kj, sj))
    return worst, info


if __name__ == '__main__':
    for N in range(2, 8):
        w, info = min_separation(N)
        print(f"N={N}: min separation {w:.5f} at {info}")
