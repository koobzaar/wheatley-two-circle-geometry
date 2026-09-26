"""Contacts between the pieces of neighbouring leaflets below the free edge (Proposition 6.2 of the note).

Facing pieces (piece 2 of leaflet k, piece 1 of leaflet k+1) meet only at their common commissure post (proved). This
script checks the other pairs numerically: at 25 heights z in [0.02, 0.98] and N = 2..7 it reports the smallest distance
between the sections of each pair of pieces of leaflets 0 and 1, away from the commissure posts (points within 0.02 of a
post are dropped). Run: uv run --project sim python sim/scripts/coaptation_pairs.py
"""
import numpy as np
from scipy.spatial import cKDTree

from wheatley import geometry as g
from wheatley import nleaflet as nl

EXCL = 0.02


def main():
    out = {}
    for N in range(2, 8):
        th = 2 * np.pi / N
        posts = [g.rot_ccw(np.array([-1.0, 0.0, 0.0]), k * th)[:2] for k in range(N)]
        for z in np.linspace(0.02, 0.98, 25):
            t = np.linspace(np.pi, float(nl.t_M(N, z)), 4000)
            L0 = {"piece1": g.surface1(t, z)[:, :2], "piece2": nl.surface2(N, t, z)[:, :2]}
            L1 = {k: g.rot_ccw(np.c_[v, np.zeros(len(v))], th)[:, :2] for k, v in L0.items()}
            for a, A in L0.items():
                for b, B in L1.items():
                    ka = np.min([np.hypot(*(A - p).T) for p in posts], axis=0) > EXCL
                    kb = np.min([np.hypot(*(B - p).T) for p in posts], axis=0) > EXCL
                    if ka.sum() and kb.sum():
                        d = cKDTree(B[kb]).query(A[ka])[0].min()
                        out[(N, a, b)] = min(out.get((N, a, b), np.inf), d)
    facing = {(N, "piece2", "piece1") for N in range(2, 8)} | {(2, "piece1", "piece2")}
    others = {k: v for k, v in out.items() if k not in facing}
    for k, v in sorted(out.items()):
        print(k, f"{v:.4f}", "(facing pair)" if k in facing else "")
    print(f"smallest distance between non-facing pieces, away from the posts: {min(others.values()):.4f}")
    return min(others.values())


if __name__ == "__main__":
    main()
